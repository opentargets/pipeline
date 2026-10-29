"""ChEMBL bioactivity dataset.

``output/chembl_bioactivity`` holds the activities with a pChEMBL value on human protein
targets that map to an Ensembl gene. Activities ChEMBL flags as targeted protein degradation
are left out.

Target components resolve to Ensembl genes as in ``drug_mechanism_of_action``.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables
from pts.schemas.chembl_bioactivity import chembl_bioactivity_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.target import chembl_accession_lookup

SCHEMA_NAME = 'public'

TABLES = {
    'activities': [
        'activity_id',
        'assay_id',
        'molregno',
        'doc_id',
        'standard_type',
        'standard_relation',
        'standard_value',
        'standard_units',
        'standard_upper_value',
        'standard_text_value',
        'pchembl_value',
        'data_validity_comment',
        'activity_comment',
        'action_type',
        'modality',
        'potential_duplicate',
    ],
    'assays': [
        'assay_id',
        'chembl_id',
        'assay_type',
        'description',
        'assay_organism',
        'relationship_type',
        'confidence_score',
        'tid',
        'variant_id',
    ],
    'ligand_eff': ['activity_id', 'le', 'lle'],
    'target_dictionary': ['tid', 'chembl_id', 'pref_name', 'target_type', 'tax_id'],
    'target_components': ['tid', 'component_id'],
    'component_sequences': ['component_id', 'accession'],
    'binding_sites': ['site_id', 'tid'],
    'site_components': ['site_id', 'component_id'],
    'molecule_dictionary': ['molregno', 'chembl_id'],
    'docs': ['doc_id', 'chembl_id', 'pubmed_id', 'doi', 'year', 'doc_type'],
    'variant_sequences': ['variant_id', 'mutation'],
}
"""Outputs are sorted by ``activityId`` and every list at its aggregation site, so read order never matters."""

FLOAT_COLUMNS = {
    'activities': ['standard_value', 'standard_upper_value', 'pchembl_value'],
    'ligand_eff': ['le', 'lle'],
}
"""Unconstrained postgres ``numeric`` columns, whose scale varies row to row: read as floats."""

DEGRADATION_MODALITY = 'Targeted Protein Degradation'
"""Activities with this modality are left out."""

HUMAN_TAX_ID = 9606

EXCLUDED_ASSAY_TYPES = ['A', 'T', 'P']
"""ADMET, toxicity and physicochemical assays: they do not measure engagement of the target."""

MEASUREMENT_CLASSES = {
    'Ki': 'affinity',
    'Kd': 'affinity',
    'IC50': 'inhibition',
    'EC50': 'functional_potency',
    'AC50': 'functional_potency',
    'XC50': 'functional_potency',
    'Potency': 'functional_potency',
    'ED50': 'functional_potency',
}
"""Every ``standard_type`` ChEMBL gives a pChEMBL value; a new one fails the step."""


def chembl_bioactivity(
    source: dict[str, Path],
    destination: dict[str, str],
    _settings: dict[str, Any],
    config: Config,
) -> None:
    """Build the ChEMBL bioactivity dataset.

    Args:
        source: Dictionary with paths to:
            - chembl: Path to the ChEMBL ``pg_dump`` archive.
            - target: Path to the target dataset.
        destination: Dictionary with paths to:
            - bioactivity: Path to write the bioactivity dataset.
        _settings: Custom settings (not used).
        config: Config object, for ``work_path``.
    """
    logger.info(f'Restoring {list(TABLES)} from {source["chembl"]}')
    tables = read_dump_tables(
        str(source['chembl']),
        TABLES,
        schema_name=SCHEMA_NAME,
        float_columns=FLOAT_COLUMNS,
        scratch_root=config.work_path,
    )

    logger.info(f'Reading target data from {source["target"]}')
    gene_df = scan_dataset(str(source['target'])).select('id', 'approvedSymbol', 'proteinIds').collect()

    bioactivity = process_bioactivity(tables, gene_df)

    logger.info(f'Writing {bioactivity.height} bioactivity rows to {destination["bioactivity"]}')
    write_dataset(bioactivity, str(destination['bioactivity']))


def process_bioactivity(tables: Mapping[str, pl.DataFrame], gene_df: pl.DataFrame) -> pl.DataFrame:
    """Build the bioactivity dataset from the ChEMBL activities.

    Args:
        tables: The raw ChEMBL tables named in `TABLES`.
        gene_df: The target dataset's `id`, `approvedSymbol` and `proteinIds`.

    Returns:
        The bioactivity rows, sorted by `activityId`.
    """
    components = _components(
        tables['target_components'],
        tables['component_sequences'],
        tables['binding_sites'],
        tables['site_components'],
        gene_df,
    )
    activities = _activities(tables['activities'], tables['assays'], tables['target_dictionary'])
    is_degradation = (pl.col('modality') == DEGRADATION_MODALITY).fill_null(False)
    logger.info(f'{activities.height:,} activities; {activities.filter(is_degradation).height:,} degradation left out')

    bioactivity = _bioactivity(activities.filter(~is_degradation), components)
    return _shape(_annotate(bioactivity, tables, components), _component_list(components), chembl_bioactivity_schema)


def _annotate(rows: pl.DataFrame, tables: Mapping[str, pl.DataFrame], components: pl.DataFrame) -> pl.DataFrame:
    """Join the per-activity annotations."""
    documents = tables['docs'].select(
        'doc_id',
        pl.col('chembl_id').alias('documentId'),
        pl.col('doc_type').alias('docType'),
        pl.col('pubmed_id').alias('pubmedId'),
        'doi',
        'year',
    )
    return (
        rows
        .join(
            tables['molecule_dictionary'].select('molregno', pl.col('chembl_id').alias('moleculeId')),
            on='molregno',
            how='left',
            validate='m:1',
        )
        .join(documents, on='doc_id', how='left', validate='m:1')
        .join(
            tables['variant_sequences'].select('variant_id', pl.col('mutation').alias('assayVariantMutation')),
            on='variant_id',
            how='left',
            validate='m:1',
        )
        # joined here, on the kept rows only: on all 24.5M activities the descriptions alone take GBs
        .join(
            tables['assays'].select(
                'assay_id',
                pl.col('description').alias('assayDescription'),
                pl.col('assay_organism').alias('assayOrganism'),
            ),
            on='assay_id',
            how='left',
            validate='m:1',
        )
        .join(tables['ligand_eff'].select('activity_id', 'le', 'lle'), on='activity_id', how='left', validate='1:1')
        .join(_complex_flags(components), on='tid', how='left', validate='m:1')
    )


def _activities(activities: pl.DataFrame, assays: pl.DataFrame, target_dictionary: pl.DataFrame) -> pl.DataFrame:
    """Every activity with the assay and target fields the prefilters need."""
    return (
        activities
        .with_columns(dup=(pl.col('potential_duplicate') == 1).fill_null(False))
        .join(
            assays.drop('description', 'assay_organism').rename({'chembl_id': 'assayId'}),
            on='assay_id',
            how='inner',
            validate='m:1',
        )
        .join(
            target_dictionary.select(
                'tid',
                'tax_id',
                pl.col('chembl_id').alias('targetChemblId'),
                pl.col('target_type').alias('targetType'),
                pl.col('pref_name').alias('targetName'),
            ),
            on='tid',
            how='left',
            validate='m:1',
        )
    )


def _components(
    target_components: pl.DataFrame,
    component_sequences: pl.DataFrame,
    binding_sites: pl.DataFrame,
    site_components: pl.DataFrame,
    gene_df: pl.DataFrame,
) -> pl.DataFrame:
    """One row per target component and Ensembl gene, or one row with no gene when unmapped.

    Returns:
        DataFrame with tid, component_id, uniprotId, targetId, approvedSymbol and inBindingSite.
    """
    lookup = (
        chembl_accession_lookup(gene_df)
        .rename({'uniprot_id': 'uniprotId', 'geneId': 'targetId'})
        .join(gene_df.select(pl.col('id').alias('targetId'), 'approvedSymbol'), on='targetId', how='left')
    )
    in_site = (
        binding_sites
        .join(site_components, on='site_id')
        .select('tid', 'component_id')
        .unique()
        .with_columns(inBindingSite=pl.lit(True))
    )
    return (
        target_components
        .join(component_sequences, on='component_id', how='inner')
        .filter(pl.col('accession').is_not_null())
        .rename({'accession': 'uniprotId'})
        .join(lookup, on='uniprotId', how='left')
        .join(in_site, on=['tid', 'component_id'], how='left')
        .with_columns(pl.col('inBindingSite').fill_null(False))
    )


def _complex_flags(components: pl.DataFrame) -> pl.DataFrame:
    """Flag multi-component targets with no binding site recorded: every subunit is credited."""
    return components.group_by('tid').agg(
        complexFlag=(pl.col('component_id').n_unique() > 1) & ~pl.col('inBindingSite').any()
    )


def _component_list(components: pl.DataFrame) -> pl.DataFrame:
    """Collect each target's components into `targetComponents`, sorted by accession then gene."""
    fields = ['uniprotId', 'targetId', 'approvedSymbol', 'inBindingSite']
    return components.group_by('tid').agg(pl.struct(fields).sort_by('uniprotId', 'targetId').alias('targetComponents'))


def _filter(frame: pl.DataFrame, label: str, condition: pl.Expr) -> pl.DataFrame:
    """Apply one prefilter and log what it removed."""
    kept = frame.filter(condition.fill_null(False))
    logger.info(f'{label}: {kept.height:,} rows left, {frame.height - kept.height:,} removed')
    return kept


def _bioactivity(activities: pl.DataFrame, components: pl.DataFrame) -> pl.DataFrame:
    """Apply the bioactivity prefilters and classify the measurements.

    Args:
        activities: Non-degradation activities from `_activities`.
        components: Target components from `_components`.

    Returns:
        The activities passing every prefilter, with `measurementClass`.

    Raises:
        ValueError: If a row with a pChEMBL value has a type missing from `MEASUREMENT_CLASSES`.
    """
    mapped_tids = components.filter(pl.col('targetId').is_not_null()).select('tid').unique()
    rows = _filter(activities, '1 human target', pl.col('tax_id') == HUMAN_TAX_ID)
    rows = _filter(rows, '2 not a potential duplicate', ~pl.col('dup'))
    rows = _filter(rows, '3 protein target type', pl.col('targetType').str.contains('PROTEIN'))
    rows = _filter(
        rows,
        '4 has an outcome',
        pl.any_horizontal(
            pl.col('standard_value', 'standard_text_value', 'activity_comment', 'pchembl_value').is_not_null()
        ),
    )
    rows = _filter(rows, '5 target-engagement assay', ~pl.col('assay_type').is_in(EXCLUDED_ASSAY_TYPES))

    unmapped = rows.join(mapped_tids, on='tid', how='anti')
    by_type = unmapped.group_by('targetType').len().sort('len', descending=True)
    logger.info(f'unmapped before filter 6, by target type: {by_type.rows()}')
    top_unmapped = (
        unmapped.group_by('targetChemblId', 'targetName').len().sort('len', 'targetChemblId', descending=[True, False])
    )
    logger.info(f'top unmapped targets: {top_unmapped.head(20).rows()}')
    rows = _filter(rows, '6 mapped to an Ensembl gene', pl.col('tid').is_in(mapped_tids['tid'].implode()))
    rows = _filter(rows, '7 has a pChEMBL value', pl.col('pchembl_value').is_not_null())

    unknown = sorted(set(rows['standard_type'].unique().to_list()) - set(MEASUREMENT_CLASSES))
    if unknown:
        raise ValueError(f'pChEMBL types missing from MEASUREMENT_CLASSES: {unknown}')

    return rows.with_columns(measurementClass=pl.col('standard_type').replace_strict(MEASUREMENT_CLASSES))


def _shape(rows: pl.DataFrame, component_lists: pl.DataFrame, schema: pl.Schema) -> pl.DataFrame:
    """Attach `targetComponents`, rename the raw columns, and keep and cast those in `schema`."""
    return (
        rows
        .join(component_lists, on='tid', how='left', validate='m:1')
        .rename({
            'activity_id': 'activityId',
            'standard_type': 'standardType',
            'standard_relation': 'standardRelation',
            'standard_value': 'standardValue',
            'standard_upper_value': 'standardUpperValue',
            'standard_units': 'standardUnits',
            'standard_text_value': 'standardTextValue',
            'pchembl_value': 'pchemblValue',
            'data_validity_comment': 'dataValidityComment',
            'activity_comment': 'activityComment',
            'action_type': 'actionType',
            'assay_type': 'assayType',
            'relationship_type': 'relationshipType',
            'confidence_score': 'confidenceScore',
        })
        .select(schema.keys())
        .cast(schema)
        .sort('activityId')
    )
