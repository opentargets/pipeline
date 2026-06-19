"""ChEMBL Molecule processing.

Processes raw ChEMBL molecule data into the Open Targets molecule format,
including synonyms, cross-references, and molecule hierarchy. Clinical-trial
(AACT) synonym mining lives in :mod:`pts.pyspark.drug_utils.aact_synonyms`.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame
from pyspark.sql.types import ArrayType, MapType, StringType

from pts.pyspark.common.session import Session
from pts.pyspark.drug_utils.aact_synonyms import merge_aact_synonyms, mine_aact_synonyms, parse_aact_batch
from pts.pyspark.drug_utils.labels import CHEMBL_SOURCE, LABEL_SOURCE_SCHEMA, as_label_source


def chembl_molecule(
    source: dict[str, str],
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Process ChEMBL molecule data.

    Args:
        source: Dictionary with paths to:
            - chembl_molecule: ChEMBL molecule JSONL
            - drugbank: Drugbank to ChEMBL ID mapping CSV
            - aact_extraction_batch_results: (optional) OpenAI batch output for
              clinical-trial synonym mining; when present, AACT synonyms are
              appended to the molecules.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='chembl_molecule', properties=properties)

    logger.info(f'Loading data from {source}')
    molecule_df = spark.load_data(source['chembl_molecule'], format='json')
    drugbank_df = spark.load_data(
        source['drugbank'],
        format='csv',
        header=True,
        sep='\t',
    )

    aact_batch_df = None
    if 'aact_extraction_batch_results' in source:
        aact_batch_df = spark.load_data(source['aact_extraction_batch_results'], format='json')

    logger.info('Processing molecules')
    output_df = process_molecules(molecule_df, drugbank_df, aact_batch_df)

    logger.info(f'Writing molecules to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_molecules(
    molecule_raw: DataFrame,
    drugbank_lookup: DataFrame,
    aact_batch: DataFrame | None = None,
) -> DataFrame:
    """Process raw ChEMBL molecule data.

    Args:
        molecule_raw: Raw ChEMBL molecule data.
        drugbank_lookup: Drugbank to ChEMBL ID mapping.
        aact_batch: (optional) OpenAI batch output for clinical-trial synonym
            mining.  When provided, AACT synonyms are appended (deduped
            case-insensitively vs existing ChEMBL labels) before the final
            name-coalesce so that AACT labels never become the molecule name.

    Returns:
        Processed molecule DataFrame.
    """
    # Prepare drugbank lookup - rename columns to match expected format
    drugbank = drugbank_lookup.select(
        f.col("From src:'1'").alias('id'),
        f.col("To src:'2'").alias('drugbank_id'),
    )

    # Preprocess molecules
    mols = _molecule_preprocess(molecule_raw, drugbank)

    # Process components
    synonyms = _process_molecule_synonyms(mols)
    cross_references = _process_molecule_cross_references(mols)
    hierarchy = _process_molecule_hierarchy(mols)

    # Combine all components
    mol_combined = (
        mols
        .drop('cross_references', 'syns')
        .join(synonyms, on='id', how='left_outer')
        .join(cross_references, on='id', how='left_outer')
        .join(hierarchy, on='id', how='left_outer')
    )

    # Optionally mine and merge AACT synonyms BEFORE the name-coalesce so that
    # AACT labels are never selected as the molecule name.
    if aact_batch is not None:
        entries = parse_aact_batch(aact_batch)
        empty_ls = f.array().cast(LABEL_SOURCE_SCHEMA)
        empty_str_arr = f.array().cast('array<string>')
        # mine_aact_synonyms / _build_chembl_indexes expect non-null arrays; coalesce here.
        mol_for_index = mol_combined.select(
            'id',
            'name',
            f.coalesce(f.col('synonyms'), empty_ls).alias('synonyms'),
            f.coalesce(f.col('tradeNames'), empty_ls).alias('tradeNames'),
            'parentId',
            f.coalesce(f.col('childChemblIds'), empty_str_arr).alias('childChemblIds'),
        )
        aact_df = mine_aact_synonyms(mol_for_index, entries)
        mol_combined = merge_aact_synonyms(mol_combined, aact_df)

    empty_label_source = f.array().cast(LABEL_SOURCE_SCHEMA)

    # Final processing - ensure name is populated and deduplicate
    return (
        mol_combined
        .withColumn('synonyms', f.coalesce(f.col('synonyms'), empty_label_source))
        .withColumn('tradeNames', f.coalesce(f.col('tradeNames'), empty_label_source))
        .withColumn(
            'name',
            f.coalesce(
                f.col('name'),
                f.element_at(
                    f.filter(f.col('synonyms'), lambda s: s['source'] == CHEMBL_SOURCE),
                    1,
                )['label'],
                f.col('id'),
            ),
        )
        .drop('drugbank_id')
        .dropDuplicates(['id'])
    )


def _molecule_preprocess(
    molecule_raw: DataFrame,
    drugbank: DataFrame,
) -> DataFrame:
    """Preprocess raw molecule data.

    Args:
        molecule_raw: Raw ChEMBL molecule data.
        drugbank: Drugbank lookup table.

    Returns:
        Preprocessed molecule DataFrame.
    """
    return (
        molecule_raw
        .select(
            f.col('molecule_chembl_id').alias('id'),
            f.col('molecule_structures.canonical_smiles').alias('canonicalSmiles'),
            f.col('molecule_structures.standard_inchi_key').alias('inchiKey'),
            # ChEMBL ships molfile as a full SD-file record (molblock + appended
            # SDF property tags). Truncate to the bare molblock by dropping
            # everything after the `M  END` terminator. If `M  END` is absent the
            # string is left unchanged.
            f.regexp_replace(
                f.col('molecule_structures.molfile'),
                r'(?s)(\nM  END\n).*',
                '$1',
            ).alias('molblock'),
            f.coalesce(f.col('molecule_type'), f.lit('Unknown')).alias('drugType'),
            f.col('pref_name').alias('name'),
            f.col('cross_references'),
            f.col('molecule_hierarchy.parent_chembl_id').alias('parentId'),
            f.col('molecule_synonyms.molecule_synonym').alias('mol_synonyms'),
            f.col('molecule_synonyms.syn_type').alias('synonym_type'),
        )
        .withColumn('syns', f.arrays_zip(f.col('mol_synonyms'), f.col('synonym_type')))
        # Remove circular references
        .withColumn(
            'parentId',
            f.when(f.col('parentId') == f.col('id'), f.lit(None)).otherwise(f.col('parentId')),
        )
        .drop('mol_synonyms', 'synonym_type')
        .join(drugbank, on='id', how='left_outer')
    )


def _process_molecule_synonyms(preprocessed_mols: DataFrame) -> DataFrame:
    """Group synonyms into sorted sets of trade names and other synonyms.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id, tradeNames, and synonyms columns ({label, source} structs).
    """
    synonyms = (
        preprocessed_mols
        .select(f.col('id'), f.explode('syns').alias('col'))
        .withColumn('syn_type', f.upper(f.col('col.synonym_type')))
        .withColumn('synonym', f.col('col.mol_synonyms'))
    )

    trade_names = (
        synonyms.filter(f.col('syn_type') == 'TRADE_NAME').groupBy('id').agg(f.collect_set('synonym').alias('_trade'))
    )

    other_synonyms = (
        synonyms.filter(f.col('syn_type') != 'TRADE_NAME').groupBy('id').agg(f.collect_set('synonym').alias('_syn'))
    )

    full = trade_names.join(other_synonyms, on='id', how='full_outer')

    return (
        full
        .withColumn(
            'synonyms',
            f.array_sort(
                f.transform(f.coalesce(f.col('_syn'), f.array()), lambda c: as_label_source(c, CHEMBL_SOURCE))
            ).cast(LABEL_SOURCE_SCHEMA),
        )
        .withColumn(
            'tradeNames',
            f.array_sort(
                f.transform(f.coalesce(f.col('_trade'), f.array()), lambda c: as_label_source(c, CHEMBL_SOURCE))
            ).cast(LABEL_SOURCE_SCHEMA),
        )
        .drop('_syn', '_trade')
    )


def _process_molecule_hierarchy(preprocessed_mols: DataFrame) -> DataFrame:
    """Group all child molecules by parent chembl_id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and childChemblIds columns.
    """
    return (
        preprocessed_mols
        .select('id', 'parentId')
        .filter(f.col('id') != f.col('parentId'))
        .filter(f.col('parentId').isNotNull())
        .groupBy('parentId')
        .agg(f.collect_set('id').alias('childChemblIds'))
        .withColumnRenamed('parentId', 'id')
    )


def _process_molecule_cross_references(preprocessed_mols: DataFrame) -> DataFrame:
    """Group cross references for each molecule id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and crossReferences columns.
    """
    chembl_xrefs = _process_chembl_cross_references(preprocessed_mols)
    drugbank_xrefs = _process_singleton_cross_references(preprocessed_mols, 'drugbank_id', 'drugbank')

    # Merge cross reference maps
    merged = _merge_cross_reference_maps(chembl_xrefs, drugbank_xrefs)
    merged = merged.filter(f.col('xref').isNotNull()).withColumnRenamed('xref', 'crossReferences')

    # Transform to array of structs format
    return (
        merged
        .select(f.col('id'), f.explode('crossReferences').alias('key', 'ids'))
        .withColumnRenamed('key', 'source')
        .groupBy('id')
        .agg(f.collect_set(f.struct(f.col('source'), f.col('ids'))).alias('crossReferences'))
    )


def _process_chembl_cross_references(preprocessed_mols: DataFrame) -> DataFrame:
    """Process ChEMBL cross references into a map structure.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and xref map columns.
    """
    chembl_xr = (
        preprocessed_mols
        .select(
            f.col('id'),
            f.explode(
                f.arrays_zip(
                    f.col('cross_references.xref_id'),
                    f.col('cross_references.xref_src'),
                )
            ).alias('sources'),
        )
        .withColumn('ref_id', f.col('sources.xref_id'))
        .withColumn('ref_src', f.col('sources.xref_src'))
        .drop('sources')
    )

    # Group by id and source to create map
    return (
        chembl_xr
        .groupBy('id', 'ref_src')
        .agg(f.collect_list('ref_id').alias('ref_ids'))
        .groupBy('id')
        .agg(f.map_from_entries(f.collect_list(f.struct('ref_src', 'ref_ids'))).alias('xref'))
    )


def _process_singleton_cross_references(
    preprocessed_mols: DataFrame,
    reference_id_column: str,
    source: str,
) -> DataFrame:
    """Process singleton cross references (e.g., drugbank_id).

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.
        reference_id_column: Column name containing the reference ID.
        source: Name of the source for the cross reference.

    Returns:
        DataFrame with id and xref map columns.
    """
    return (
        preprocessed_mols
        .filter(f.col(reference_id_column).isNotNull())
        .select(f.col('id'), f.col(reference_id_column).cast('string'))
        .groupBy('id')
        .agg(f.collect_set(reference_id_column).alias(reference_id_column))
        .withColumn('xref', f.create_map(f.lit(source), f.col(reference_id_column)))
        .drop(reference_id_column)
    )


def _merge_cross_reference_maps(ref1: DataFrame, ref2: DataFrame) -> DataFrame:
    """Merge two cross reference map DataFrames.

    Args:
        ref1: First DataFrame with id and xref columns.
        ref2: Second DataFrame with id and xref columns.

    Returns:
        Merged DataFrame with combined xref maps.
    """
    empty_map = f.create_map().cast(MapType(StringType(), ArrayType(StringType())))

    r1 = ref1.select(f.col('id'), f.coalesce(f.col('xref'), empty_map).alias('x'))
    r2 = ref2.select(f.col('id'), f.coalesce(f.col('xref'), empty_map).alias('y'))

    return (
        r1
        .join(r2, on='id', how='full_outer')
        .select(
            f.col('id'),
            f.coalesce(f.col('x'), empty_map).alias('x'),
            f.coalesce(f.col('y'), empty_map).alias('y'),
        )
        .withColumn('xref', f.map_concat(f.col('x'), f.col('y')))
        .drop('x', 'y')
    )
