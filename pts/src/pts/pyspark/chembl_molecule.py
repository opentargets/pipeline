"""ChEMBL Molecule processing.

Processes raw ChEMBL molecule tables into the Open Targets molecule format,
including synonyms, DrugBank cross-references, and molecule hierarchy. ChEMBL's
own cross-reference table is deliberately not joined here (see
:func:`_process_molecule_cross_references`). Clinical-trial (AACT) synonym
mining lives in :mod:`pts.pyspark.drug_utils.aact_synonyms`.
"""

from typing import Any

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

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
            - molecule_dictionary: Raw ChEMBL molecule_dictionary parquet.
            - compound_structures: Raw ChEMBL compound_structures parquet.
            - molecule_hierarchy: Raw ChEMBL molecule_hierarchy parquet.
            - molecule_synonyms: Raw ChEMBL molecule_synonyms parquet.
            - drugbank: Drugbank to ChEMBL ID mapping CSV.
            - aact_extraction_batch_results: (optional) OpenAI batch output for
              clinical-trial synonym mining; when present, AACT synonyms are
              appended to the molecules.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='chembl_molecule', properties=properties)

    logger.info(f'Loading data from {source}')
    molecule_dictionary = spark.load_data(source['molecule_dictionary'])
    compound_structures = spark.load_data(source['compound_structures'])
    molecule_hierarchy = spark.load_data(source['molecule_hierarchy'])
    molecule_synonyms = spark.load_data(source['molecule_synonyms'])
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
    output_df = process_molecules(
        molecule_dictionary,
        compound_structures,
        molecule_hierarchy,
        molecule_synonyms,
        drugbank_df,
        aact_batch_df,
    )

    logger.info(f'Writing molecules to {destination}')
    output_df.write.parquet(destination, mode='overwrite')


def process_molecules(
    molecule_dictionary: DataFrame,
    compound_structures: DataFrame,
    molecule_hierarchy: DataFrame,
    molecule_synonyms: DataFrame,
    drugbank_lookup: DataFrame,
    aact_batch: DataFrame | None = None,
) -> DataFrame:
    """Build ChEMBL molecules by joining raw ChEMBL tables.

    Args:
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        compound_structures: Raw ChEMBL compound_structures table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        molecule_synonyms: Raw ChEMBL molecule_synonyms table.
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
    mols = _molecule_preprocess(
        molecule_dictionary,
        compound_structures,
        molecule_hierarchy,
        molecule_synonyms,
        drugbank,
    )

    # Process components
    synonyms = _process_molecule_synonyms(mols)
    cross_references = _process_molecule_cross_references(mols)
    hierarchy = _process_molecule_hierarchy(mols)

    # Combine all components
    mol_combined = (
        mols
        .drop('syns')
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
    molecule_dictionary: DataFrame,
    compound_structures: DataFrame,
    molecule_hierarchy: DataFrame,
    molecule_synonyms: DataFrame,
    drugbank: DataFrame,
) -> DataFrame:
    """Preprocess raw ChEMBL molecule tables into one row per molecule.

    Args:
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        compound_structures: Raw ChEMBL compound_structures table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        molecule_synonyms: Raw ChEMBL molecule_synonyms table.
        drugbank: Drugbank lookup table (id, drugbank_id).

    Returns:
        Preprocessed molecule DataFrame.
    """
    parent = (
        molecule_hierarchy
        .join(
            molecule_dictionary.select(
                f.col('molregno').alias('parent_molregno'),
                f.col('chembl_id').alias('parentId'),
            ),
            on='parent_molregno',
            how='left',
        )
        .select('molregno', 'parentId')
    )

    # One struct array per molecule, ordered by molsyn_id: replaces the old
    # arrays_zip of two parallel nested arrays.
    synonyms = (
        molecule_synonyms
        .groupBy('molregno')
        .agg(f.sort_array(f.collect_list(f.struct('molsyn_id', 'synonyms', 'syn_type'))).alias('syns'))
    )

    return (
        molecule_dictionary
        .withColumnRenamed('chembl_id', 'id')
        .join(compound_structures, on='molregno', how='left_outer')
        .join(parent, on='molregno', how='left_outer')
        .join(synonyms, on='molregno', how='left_outer')
        .select(
            'id',
            f.col('canonical_smiles').alias('canonicalSmiles'),
            f.col('standard_inchi_key').alias('inchiKey'),
            # ChEMBL ships some molfile values as a full SD-file record (molblock
            # + appended SDF property tags); truncate to the bare molblock by
            # dropping everything after the `M  END` terminator. If `M  END` is
            # absent the string is left unchanged.
            f.regexp_replace(
                f.col('molfile'),
                # the terminator may or may not be followed by a newline: the
                # relational column is inconsistent, unlike the ES value which
                # always had one before its SDF property block
                r'(?s)(\nM  END)\n?.*',
                '$1',
            ).alias('molblock'),
            f.coalesce(f.col('molecule_type'), f.lit('Unknown')).alias('drugType'),
            f.trim(f.col('pref_name')).alias('name'),
            'parentId',
            'syns',
        )
        # Remove circular references
        .withColumn(
            'parentId',
            f.when(f.col('parentId') == f.col('id'), f.lit(None)).otherwise(f.col('parentId')),
        )
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
        .withColumn('syn_type', f.upper(f.col('col.syn_type')))
        .withColumn('synonym', f.col('col.synonyms'))
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
    """Group DrugBank cross references for each molecule id.

    ChEMBL's own cross-reference table is not joined here. Rebuilding it from
    the raw relational dump is unreliable (the best measured rule reaches 44%
    precision), so it was dropped rather than fabricated. DrugBank is
    unaffected, since it comes from a separate input.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and crossReferences columns.
    """
    drugbank_xrefs = _process_singleton_cross_references(preprocessed_mols, 'drugbank_id', 'drugbank')

    return (
        drugbank_xrefs
        .filter(f.col('xref').isNotNull())
        .select(f.col('id'), f.explode('xref').alias('key', 'ids'))
        .withColumnRenamed('key', 'source')
        .groupBy('id')
        .agg(f.collect_set(f.struct(f.col('source'), f.col('ids'))).alias('crossReferences'))
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
