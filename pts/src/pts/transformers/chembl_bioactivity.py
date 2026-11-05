"""ChEMBL bioactivity dataset.

``output/chembl_bioactivity`` holds the activities with a pChEMBL value on human protein
targets that map to an Ensembl gene, one row per activity and Open Targets target it credits.
Activities ChEMBL flags as targeted protein degradation are left out.

Target components resolve to Ensembl genes as in ``drug_mechanism_of_action``. An activity
credits the genes of its target's binding-site components where ChEMBL records a site on a
mapped component, and every mapped gene otherwise; ``complexFlag`` marks activities that credit
more than one gene. ChEMBL's own target name and type are not published.

Every activity carries the ChEMBL id of its molecule (``drugFromSourceId``) and, where the
molecule or its parent is in the drug molecule dataset, the parent's ``drugId``. The action type
ChEMBL records on the activity is ``assayActionType``; ``actionTypes`` are those of the curated
mechanisms of action of the same drug on the row's target.

``activityOutcome`` is the depositor's verdict, where one is recorded: a fitted potency, and so
a pChEMBL value, can come from a curve the depositor judged inactive or inconclusive.
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
        'standard_value',
        'standard_units',
        'standard_text_value',
        'pchembl_value',
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
    'docs': ['doc_id', 'chembl_id', 'pubmed_id', 'year', 'doc_type'],
    'variant_sequences': ['variant_id', 'mutation'],
}
"""The ChEMBL tables and columns the step reads."""

FLOAT_COLUMNS = {
    'activities': ['standard_value', 'pchembl_value'],
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

PCHEMBL_UNITS = 'nM'
"""The unit of every standard value with a pChEMBL value; another unit fails the step."""

PLACEHOLDER_MUTATIONS = ['UNDEFINED MUTATION']
"""ChEMBL ``variant_sequences.mutation`` values that name no mutation: published as null."""

ACTIVITY_OUTCOMES = {
    'active': 'active',
    'not active': 'inactive',
    'inactive': 'inactive',
    'inconclusive': 'inconclusive',
}
"""The outcomes depositors record in ``activity_comment``, lowercased: mostly on PubChem BioAssay
screens, also on a few other datasets. Any other comment (free text from publications,
PubChem's ``Unspecified``) gives no outcome."""


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
            - drug_molecule: Path to the drug molecule dataset.
            - mechanism_of_action: Path to the drug mechanism of action dataset.
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
    gene_df = scan_dataset(str(source['target'])).select('id', 'proteinIds').collect()
    logger.info(f'Reading drugs from {source["drug_molecule"]} and {source["mechanism_of_action"]}')
    drug_df = scan_dataset(str(source['drug_molecule'])).select('id', 'parentId', 'childChemblIds').collect()
    moa_df = scan_dataset(str(source['mechanism_of_action'])).select('actionType', 'chemblIds', 'targets').collect()

    bioactivity = process_bioactivity(tables, gene_df, drug_df, moa_df)

    logger.info(f'Writing {bioactivity.height} bioactivity rows to {destination["bioactivity"]}')
    write_dataset(bioactivity, str(destination['bioactivity']))


def process_bioactivity(
    tables: Mapping[str, pl.DataFrame],
    gene_df: pl.DataFrame,
    drug_df: pl.DataFrame,
    moa_df: pl.DataFrame,
) -> pl.DataFrame:
    """Build the bioactivity dataset from the ChEMBL activities.

    Args:
        tables: The raw ChEMBL tables named in `TABLES`.
        gene_df: The target dataset's `id` and `proteinIds`.
        drug_df: The drug molecule dataset's `id`, `parentId` and `childChemblIds`.
        moa_df: The mechanism of action dataset's `actionType`, `chemblIds` and `targets`.

    Returns:
        The bioactivity rows, one per activity and credited target, sorted by `activityId`
        then `targetId`. Every list is sorted where it is aggregated, so the order tables
        are read in never reaches the output.
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

    credited = _credited_targets(components)
    bioactivity = _bioactivity(activities.filter(~is_degradation), credited['tid'].unique())
    rows = bioactivity.join(credited, on='tid', how='inner', validate='m:m')
    return _shape(_annotate(rows, tables, _drug_ids(drug_df), _mechanism_pairs(moa_df)))


def _annotate(
    rows: pl.DataFrame,
    tables: Mapping[str, pl.DataFrame],
    drugs: pl.DataFrame,
    mechanisms: pl.DataFrame,
) -> pl.DataFrame:
    """Join the per-activity annotations, and the mechanisms of action per row's target."""
    documents = tables['docs'].select(
        'doc_id',
        pl.col('chembl_id').alias('documentId'),
        pl.col('doc_type').alias('docType'),
        pl.col('pubmed_id').alias('pubmedId'),
        'year',
    )
    return (
        rows
        .join(
            tables['molecule_dictionary'].select('molregno', pl.col('chembl_id').alias('drugFromSourceId')),
            on='molregno',
            how='left',
            validate='m:1',
        )
        .join(drugs, on='drugFromSourceId', how='left', validate='m:1')
        .pipe(_with_action_types, mechanisms)
        .join(documents, on='doc_id', how='left', validate='m:1')
        .join(
            tables['variant_sequences'].select(
                'variant_id',
                pl
                .when(~pl.col('mutation').is_in(PLACEHOLDER_MUTATIONS))
                .then(pl.col('mutation'))
                .alias('variantFromSource'),
            ),
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
        .join(tables['ligand_eff'].select('activity_id', 'le', 'lle'), on='activity_id', how='left', validate='m:1')
    )


def _activities(activities: pl.DataFrame, assays: pl.DataFrame, target_dictionary: pl.DataFrame) -> pl.DataFrame:
    """Every activity with the assay and target fields the prefilters need."""
    return activities.join(
        assays.drop('description', 'assay_organism').rename({'chembl_id': 'assayId'}),
        on='assay_id',
        how='inner',
        validate='m:1',
    ).join(
        target_dictionary.select(
            'tid',
            'tax_id',
            pl.col('chembl_id').alias('targetFromSourceId'),
            pl.col('target_type').alias('targetType'),
            pl.col('pref_name').alias('targetName'),
        ),
        on='tid',
        how='left',
        validate='m:1',
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
        DataFrame with tid, component_id, uniprotId, targetId and inBindingSite.
    """
    lookup = chembl_accession_lookup(gene_df).rename({'uniprot_id': 'uniprotId', 'geneId': 'targetId'})
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


def _credited_targets(components: pl.DataFrame) -> pl.DataFrame:
    """The genes each ChEMBL target credits, and whether it credits more than one.

    A target credits the genes of its binding-site components where a site is recorded on a
    mapped component, and every mapped gene otherwise. A target with no mapped component
    credits none.

    Returns:
        DataFrame with tid, targetId and complexFlag, one row per target and credited gene.
    """
    mapped = components.filter(pl.col('targetId').is_not_null())
    credited = mapped.filter(pl.col('inBindingSite') | ~pl.col('inBindingSite').any().over('tid'))
    return credited.select('tid', 'targetId').unique().with_columns(complexFlag=pl.len().over('tid') > 1)


def _drug_ids(drug_df: pl.DataFrame) -> pl.DataFrame:
    """Map each ChEMBL molecule id in the drug molecule dataset, or listed as a child of one, to its parent drug.

    The parent is used only where it is itself in the drug molecule dataset, so every `drugId`
    resolves there; otherwise the molecule maps to its own drug.

    Returns:
        DataFrame with drugFromSourceId and drugId, one row per drugFromSourceId.

    Raises:
        ValueError: If a molecule id resolves to more than one drug.
    """
    drug_id = (
        pl
        .when(pl.col('parentId').is_in(drug_df['id'].implode()))
        .then(pl.col('parentId'))
        .otherwise(pl.col('id'))
        .alias('drugId')
    )
    pairs = pl.concat([
        drug_df.select(pl.col('id').alias('drugFromSourceId'), drug_id),
        drug_df
        .select(pl.col('childChemblIds').alias('drugFromSourceId'), drug_id)
        .explode('drugFromSourceId')
        .drop_nulls('drugFromSourceId'),
    ]).unique()
    ambiguous = pairs.filter(pl.col('drugFromSourceId').is_duplicated())
    if ambiguous.height:
        raise ValueError(f'molecules resolving to several drugs: {ambiguous.sort(pl.all()).head(10).rows()}')
    return pairs


def _mechanism_pairs(moa_df: pl.DataFrame) -> pl.DataFrame:
    """One row per distinct (molecule id, target gene, action type) in the mechanisms of action."""
    return (
        moa_df
        .select(pl.col('chemblIds').alias('drug'), pl.col('targets').alias('targetId'), 'actionType')
        .explode('drug')
        .explode('targetId')
        .drop_nulls()
        .unique()
    )


def _with_action_types(rows: pl.DataFrame, mechanisms: pl.DataFrame) -> pl.DataFrame:
    """Attach `actionTypes`: those of the mechanisms of the same drug on the row's target.

    The drug matches on the molecule's own id or its parent `drugId`, as mechanisms list both.
    The list is sorted, and empty where no mechanism matches.
    """
    keys = pl.concat([
        rows.select('activity_id', 'targetId', pl.col('drugFromSourceId').alias('drug')),
        rows.select('activity_id', 'targetId', pl.col('drugId').alias('drug')),
    ]).drop_nulls('drug')
    action_types = (
        keys
        .join(mechanisms, on=['drug', 'targetId'])
        .group_by('activity_id', 'targetId')
        .agg(pl.col('actionType').unique().sort().alias('actionTypes'))
    )
    return rows.join(action_types, on=['activity_id', 'targetId'], how='left', validate='1:1').with_columns(
        pl.col('actionTypes').fill_null([])
    )


def _filter(frame: pl.DataFrame, label: str, condition: pl.Expr) -> pl.DataFrame:
    """Apply one prefilter and log what it removed."""
    kept = frame.filter(condition.fill_null(False))
    logger.info(f'{label}: {kept.height:,} rows left, {frame.height - kept.height:,} removed')
    return kept


def _bioactivity(activities: pl.DataFrame, mapped_tids: pl.Series) -> pl.DataFrame:
    """Apply the bioactivity prefilters and classify the measurements.

    Args:
        activities: Non-degradation activities from `_activities`.
        mapped_tids: The targets with at least one component mapped to a gene.

    Returns:
        The activities passing every prefilter, with `measurementClass` and `activityOutcome`.

    Raises:
        ValueError: If a row with a pChEMBL value has a type missing from `MEASUREMENT_CLASSES`,
            or a unit other than `PCHEMBL_UNITS`.
    """
    is_mapped = pl.col('tid').is_in(mapped_tids.implode())
    rows = _filter(activities, '1 human target', pl.col('tax_id') == HUMAN_TAX_ID)
    rows = _filter(rows, '2 not a potential duplicate', pl.col('potential_duplicate').ne_missing(1))
    rows = _filter(rows, '3 protein target type', pl.col('targetType').str.contains('PROTEIN'))
    # implied by filter 7 (pchembl_value is one of the outcomes); kept so the logged funnel
    # matches the confirmed prefilter list
    rows = _filter(
        rows,
        '4 has an outcome',
        pl.any_horizontal(
            pl.col('standard_value', 'standard_text_value', 'activity_comment', 'pchembl_value').is_not_null()
        ),
    )
    rows = _filter(rows, '5 target-engagement assay', ~pl.col('assay_type').is_in(EXCLUDED_ASSAY_TYPES))

    unmapped = rows.filter(~is_mapped)
    by_type = unmapped.group_by('targetType').len().sort('len', descending=True)
    logger.info(f'unmapped before filter 6, by target type: {by_type.rows()}')
    top_unmapped = (
        unmapped
        .group_by('targetFromSourceId', 'targetName')
        .len()
        .sort('len', 'targetFromSourceId', descending=[True, False])
    )
    logger.info(f'top unmapped targets: {top_unmapped.head(20).rows()}')
    rows = _filter(rows, '6 mapped to an Ensembl gene', is_mapped)
    rows = _filter(rows, '7 has a pChEMBL value', pl.col('pchembl_value').is_not_null())

    unknown = sorted(set(rows['standard_type'].unique().to_list()) - set(MEASUREMENT_CLASSES))
    if unknown:
        raise ValueError(f'pChEMBL types missing from MEASUREMENT_CLASSES: {unknown}')
    units = sorted(set(rows['standard_units'].unique().to_list()) - {PCHEMBL_UNITS})
    if units:
        raise ValueError(f'pChEMBL rows with units other than {PCHEMBL_UNITS}: {units}')

    return rows.with_columns(
        measurementClass=pl.col('standard_type').replace_strict(MEASUREMENT_CLASSES),
        activityOutcome=pl
        .col('activity_comment')
        .str.strip_chars()
        .str.to_lowercase()
        .replace_strict(ACTIVITY_OUTCOMES, default=None),
    )


def _shape(rows: pl.DataFrame) -> pl.DataFrame:
    """Rename the raw columns, and keep and cast the schema's."""
    return (
        rows
        .rename({
            'activity_id': 'activityId',
            'standard_type': 'standardMeasurementType',
            'standard_value': 'standardMeasurementValue',
            'pchembl_value': 'pchemblValue',
            'action_type': 'assayActionType',
            'assay_type': 'assayType',
            'confidence_score': 'confidenceScore',
        })
        .select(chembl_bioactivity_schema.keys())
        .cast(chembl_bioactivity_schema)
        .sort('activityId', 'targetId')
    )
