"""ChEMBL Molecule processing.

Builds the Open Targets molecule format from the raw ChEMBL tables. Synonyms and trade
names come from ``molecule_synonyms``, cross-references from DrugBank, and clinical-trial
synonyms from :mod:`pts.transformers.utils.aact_synonyms`. PAINS structural alerts are added
for every molecule with a structure; ``drug_molecule`` does not read them.
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.schemas.drug_molecule import structural_alert_schema
from pts.transformers.clinical_mira.staged_outputs import EXTRACTION
from pts.transformers.utils.aact_synonyms import merge_aact_synonyms, mine_aact_synonyms, parse_aact_entries
from pts.transformers.utils.dataset import scan_dataset, write_dataset

SCHEMA_NAME = 'public'

TABLES = {
    'molecule_dictionary': ['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
    'compound_structures': ['molregno', 'canonical_smiles', 'standard_inchi_key', 'molfile'],
    'molecule_hierarchy': ['molregno', 'parent_molregno'],
    'molecule_synonyms': ['molsyn_id', 'molregno', 'synonyms', 'syn_type'],
    'compound_structural_alerts': ['molregno', 'alert_id'],
    'structural_alerts': ['alert_id', 'alert_set_id', 'alert_name'],
    'structural_alert_sets': ['alert_set_id', 'set_name'],
}
"""No ``order_by``: every published list sorts at its aggregation site instead."""

CHEMBL_SOURCE = 'ChEMBL'

PAINS_ALERT_SET = 'PAINS'
"""ChEMBL's name for the pan-assay interference set (Baell and Holloway, J Med Chem 2010)."""


def chembl_molecule(
    source: dict[str, Path],
    destination: Path,
    _settings: dict[str, Any],
    config: Config,
) -> None:
    """Transform raw ChEMBL molecule tables into the Open Targets molecule format.

    Args:
        source: Dictionary with paths to:
            - chembl: Path to the ChEMBL ``pg_dump`` archive.
            - drugbank: DrugBank-to-ChEMBL id mapping (tab-separated, ``.csv.gz``).
            - aact_trial_extraction: (optional) extracted AACT trial results for
              clinical-trial synonym mining; when present, AACT synonyms are
              appended to the molecules.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        config: Config object, for ``work_path``.
    """
    aact_batch = None
    if 'aact_trial_extraction' in source:
        logger.info(f'Reading AACT extraction from {source["aact_trial_extraction"]}')
        aact_batch = EXTRACTION.read(str(source['aact_trial_extraction']))

    logger.info(f'Restoring {list(TABLES)} from {source["chembl"]}')
    tables = read_dump_tables(str(source['chembl']), TABLES, schema_name=SCHEMA_NAME, scratch_root=config.work_path)

    logger.info(f'Reading drugbank lookup from {source["drugbank"]}')
    drugbank_lookup = scan_dataset(str(source['drugbank']), format='tsv', has_header=True).collect()

    logger.info('Processing molecules')
    output_df = process_molecules(
        tables['molecule_dictionary'],
        tables['compound_structures'],
        tables['molecule_hierarchy'],
        tables['molecule_synonyms'],
        drugbank_lookup,
        pains_alerts(
            tables['compound_structural_alerts'], tables['structural_alerts'], tables['structural_alert_sets']
        ),
        aact_batch,
    )

    logger.info(f'Writing molecules to {destination}')
    write_dataset(output_df, str(destination))


def process_molecules(
    molecule_dictionary: pl.DataFrame,
    compound_structures: pl.DataFrame,
    molecule_hierarchy: pl.DataFrame,
    molecule_synonyms: pl.DataFrame,
    drugbank_lookup: pl.DataFrame,
    structural_alerts: pl.DataFrame,
    aact_batch: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build ChEMBL molecules by joining raw ChEMBL tables.

    Args:
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        compound_structures: Raw ChEMBL compound_structures table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        molecule_synonyms: Raw ChEMBL molecule_synonyms table.
        drugbank_lookup: Drugbank to ChEMBL id mapping, as read from the raw file.
        structural_alerts: The PAINS alerts per molregno, from :func:`pains_alerts`.
        aact_batch: (optional) AACT trial extractions, as published by
            :py:mod:`pts.transformers.clinical_mira.aact_trial_extraction`.
            When provided, AACT synonyms are appended (deduped case-insensitively vs
            existing ChEMBL labels) before the final name-coalesce so that AACT labels
            never become the molecule name.

    Returns:
        Processed molecule DataFrame with the columns of
        :data:`pts.schemas.drug_molecule.chembl_molecule_schema`.
    """
    # Prepare drugbank lookup -- rename columns to match expected format
    drugbank = drugbank_lookup.select(
        pl.col("From src:'1'").alias('id'),
        pl.col("To src:'2'").alias('drugbank_id'),
    )

    mols = _molecule_preprocess(
        molecule_dictionary,
        compound_structures,
        molecule_hierarchy,
        molecule_synonyms,
        drugbank,
    )

    synonyms = _process_molecule_synonyms(mols)
    cross_references = _process_molecule_cross_references(mols)
    hierarchy = _process_molecule_hierarchy(mols)

    mol_combined = (
        mols
        .drop('syns')
        .join(synonyms, on='id', how='left')
        .join(cross_references, on='id', how='left')
        .join(hierarchy, on='id', how='left')
        .join(structural_alerts, on='molregno', how='left', validate='m:1')
    )

    # Optionally mine and merge AACT synonyms BEFORE the name-coalesce so that
    # AACT labels are never selected as the molecule name.
    if aact_batch is not None:
        entries = parse_aact_entries(aact_batch)
        # `mine_aact_synonyms` documents a non-null contract on these list columns. The
        # fills satisfy it explicitly rather than relying on the mining internals to
        # tolerate nulls.
        mol_for_index = mol_combined.select(
            'id',
            'name',
            pl.col('synonyms').fill_null([]),
            pl.col('tradeNames').fill_null([]),
            'parentId',
            pl.col('childChemblIds').fill_null([]),
        )
        aact_df = mine_aact_synonyms(mol_for_index, entries)
        mol_combined = merge_aact_synonyms(mol_combined, aact_df)

    # Final processing -- ensure name is populated and deduplicate
    return (
        mol_combined
        .with_columns(
            synonyms=pl.col('synonyms').fill_null([]),
            tradeNames=pl.col('tradeNames').fill_null([]),
            # no alert means none matched, but only on a structure ChEMBL could check
            structuralAlerts=pl
            .when(
                pl.col('structuralAlerts').is_null()
                & pl.all_horizontal(pl.col('canonicalSmiles', 'molblock').is_null())
            )
            .then(None)
            .otherwise(pl.col('structuralAlerts').fill_null([]))
            .cast(structural_alert_schema),
        )
        .with_columns(
            name=pl.coalesce(
                pl.col('name'),
                pl
                .col('synonyms')
                .list.eval(pl.element().filter(pl.element().struct.field('source') == CHEMBL_SOURCE))
                .list.first()
                .struct.field('label'),
                pl.col('id'),
            )
        )
        .unique(subset=['id'], maintain_order=True)
        .select(
            'id',
            'canonicalSmiles',
            'inchiKey',
            'molblock',
            'drugType',
            'name',
            'parentId',
            'childChemblIds',
            'synonyms',
            'tradeNames',
            'crossReferences',
            'structuralAlerts',
        )
    )


def pains_alerts(
    compound_structural_alerts: pl.DataFrame,
    structural_alerts: pl.DataFrame,
    structural_alert_sets: pl.DataFrame,
) -> pl.DataFrame:
    """Collect the PAINS alerts of each molecule that has any; alerts of the other ChEMBL sets are left out.

    Args:
        compound_structural_alerts: Raw ChEMBL compound_structural_alerts table.
        structural_alerts: Raw ChEMBL structural_alerts table.
        structural_alert_sets: Raw ChEMBL structural_alert_sets table.

    Returns:
        DataFrame with molregno and structuralAlerts, sorted by alert id.

    Raises:
        ValueError: If no alert set, or more than one, is named `PAINS_ALERT_SET`.
    """
    # matched by name: a renamed set in a later release must fail, not empty the column
    pains = structural_alert_sets.filter(pl.col('set_name') == PAINS_ALERT_SET)
    if pains.height != 1:
        raise ValueError(f'expected one alert set named {PAINS_ALERT_SET}, found {pains.height}')

    return (
        compound_structural_alerts
        .select('molregno', 'alert_id')
        .unique()
        .join(structural_alerts.join(pains, on='alert_set_id', how='semi'), on='alert_id', how='inner')
        .group_by('molregno')
        .agg(
            pl
            .struct(pl.col('alert_id').alias('alertId'), pl.col('alert_name').alias('alertName'))
            .sort_by('alert_id')
            .alias('structuralAlerts')
        )
    )


def _molecule_preprocess(
    molecule_dictionary: pl.DataFrame,
    compound_structures: pl.DataFrame,
    molecule_hierarchy: pl.DataFrame,
    molecule_synonyms: pl.DataFrame,
    drugbank: pl.DataFrame,
) -> pl.DataFrame:
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
    parent = molecule_hierarchy.join(
        molecule_dictionary.select(
            pl.col('molregno').alias('parent_molregno'),
            pl.col('chembl_id').alias('parentId'),
        ),
        on='parent_molregno',
        how='left',
    ).select('molregno', 'parentId')

    synonyms = molecule_synonyms.group_by('molregno').agg(pl.struct('molsyn_id', 'synonyms', 'syn_type').alias('syns'))

    return (
        molecule_dictionary
        .rename({'chembl_id': 'id'})
        .join(compound_structures, on='molregno', how='left')
        .join(parent, on='molregno', how='left')
        .join(synonyms, on='molregno', how='left')
        .select(
            'molregno',
            'id',
            pl.col('canonical_smiles').alias('canonicalSmiles'),
            pl.col('standard_inchi_key').alias('inchiKey'),
            # ChEMBL ships some molfile values as a full SD-file record (molblock +
            # appended SDF property tags); truncate to the bare molblock by dropping
            # everything after the `M  END` terminator. If `M  END` is absent the
            # string is left unchanged.
            pl
            .col('molfile')
            .str.replace_all(
                # the terminator is inconsistently followed by a newline in the source
                # column. Emit exactly one either way -- the published column carries a
                # trailing newline, so dropping it would change every molblock.
                r'(?s)(\nM  END)\n?.*',
                '$1\n',
            )
            .alias('molblock'),
            pl.col('molecule_type').fill_null('Unknown').alias('drugType'),
            # `pref_name` carries trailing spaces that must not reach the published
            # `name`. Pinned to ' ' rather than left bare because `str.strip_chars()`
            # with no argument strips all Unicode whitespace, which is wider than the
            # padding this column actually has -- and wider than what the published
            # values were produced with.
            pl.col('pref_name').str.strip_chars(' ').alias('name'),
            'parentId',
            'syns',
        )
        # Remove circular references
        .with_columns(parentId=pl.when(pl.col('parentId') == pl.col('id')).then(None).otherwise(pl.col('parentId')))
        .join(drugbank, on='id', how='left')
    )


def _process_molecule_synonyms(preprocessed_mols: pl.DataFrame) -> pl.DataFrame:
    """Group synonyms into sets of trade names and other synonyms.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id, tradeNames, and synonyms columns ({label, source} structs).
    """
    synonyms = (
        preprocessed_mols
        .select('id', 'syns')
        .explode('syns', empty_as_null=True)
        # A molecule with no molecule_synonyms rows joins to a null `syns` array, which
        # polars' explode keeps as a null row rather than discarding.
        .drop_nulls('syns')
        .unnest('syns')
        .with_columns(
            syn_type=pl.col('syn_type').str.to_uppercase(),
            synonym=pl.col('synonyms'),
        )
    )

    trade_names = (
        synonyms
        .filter(pl.col('syn_type') == 'TRADE_NAME')
        .group_by('id')
        .agg(pl.col('synonym').drop_nulls().unique().alias('_trade'))
    )

    other_synonyms = (
        synonyms
        .filter(pl.col('syn_type') != 'TRADE_NAME')
        .group_by('id')
        .agg(pl.col('synonym').drop_nulls().unique().alias('_syn'))
    )

    full = trade_names.join(other_synonyms, on='id', how='full', coalesce=True)

    return full.with_columns(
        synonyms=pl
        .col('_syn')
        .fill_null([])
        .list.eval(pl.struct(pl.element().alias('label'), pl.lit(CHEMBL_SOURCE).alias('source')))
        .list.sort(),
        tradeNames=pl
        .col('_trade')
        .fill_null([])
        .list.eval(pl.struct(pl.element().alias('label'), pl.lit(CHEMBL_SOURCE).alias('source')))
        .list.sort(),
    ).drop('_syn', '_trade')


def _process_molecule_hierarchy(preprocessed_mols: pl.DataFrame) -> pl.DataFrame:
    """Group all child molecules by parent chembl_id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and childChemblIds columns.
    """
    return (
        preprocessed_mols
        .select('id', 'parentId')
        .filter(pl.col('id') != pl.col('parentId'))
        .filter(pl.col('parentId').is_not_null())
        .group_by('parentId')
        .agg(pl.col('id').drop_nulls().unique().sort().alias('childChemblIds'))
        .rename({'parentId': 'id'})
    )


def _process_molecule_cross_references(preprocessed_mols: pl.DataFrame) -> pl.DataFrame:
    """Group DrugBank cross references for each molecule id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and crossReferences columns.
    """
    return _process_singleton_cross_references(preprocessed_mols, 'drugbank_id', 'drugbank')


def _process_singleton_cross_references(
    preprocessed_mols: pl.DataFrame,
    reference_id_column: str,
    source: str,
) -> pl.DataFrame:
    """Process singleton cross references (e.g., drugbank_id) into a crossReferences column.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.
        reference_id_column: Column name containing the reference id.
        source: Name of the source for the cross reference.

    Returns:
        DataFrame with id and crossReferences (array<struct<source,ids>>) columns.
    """
    return (
        preprocessed_mols
        .filter(pl.col(reference_id_column).is_not_null())
        .select('id', pl.col(reference_id_column).cast(pl.Utf8))
        .group_by('id')
        .agg(pl.col(reference_id_column).drop_nulls().unique().sort().alias('ids'))
        .with_columns(crossReferences=pl.concat_list(pl.struct(pl.lit(source).alias('source'), pl.col('ids'))))
        .select('id', 'crossReferences')
    )
