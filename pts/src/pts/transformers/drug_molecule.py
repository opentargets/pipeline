"""Drug index generation.

Combines molecule data with clinical reports, mechanisms of action, and chemical probes
to produce the final drug index. Filters to include only molecules that qualify as
"drugs" and generates human-readable descriptions.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from clinical_mining.dataset.clinical_indication import (
    CATEGORY_RANKS_STR,
    RANK_TO_CATEGORY_STR,
    ClinicalStageCategory,
)
from loguru import logger
from otter.config.model import Config

from pts.schemas.drug_molecule import drug_molecule_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset

APPROVED_STAGE_CODE = ClinicalStageCategory.APPROVAL.value

STAGE_FOR_MAX_MAPPING = {
    ClinicalStageCategory.WITHDRAWAL.value: APPROVED_STAGE_CODE,
    ClinicalStageCategory.PHASE_4.value: APPROVED_STAGE_CODE,
}
_DEFAULT_STAGE_RANK_VALUE = CATEGORY_RANKS_STR[ClinicalStageCategory.UNKNOWN.value]
_DEFAULT_STAGE_NAME_VALUE = RANK_TO_CATEGORY_STR[_DEFAULT_STAGE_RANK_VALUE]

# the source string as chembl_molecule writes it, matched exactly by the is_drug filter
DRUGBANK_SOURCE = 'drugbank'
PROBES_AND_DRUGS_SOURCE = 'Probes&Drugs'

# sorted because `group_by` does not promise row order
_INDICATION_SORT = 'diseaseId'


def _stage_rank(stage: pl.Expr) -> pl.Expr:
    """Rank a clinical stage, folding WITHDRAWAL and PHASE_4 into APPROVAL.

    A lower rank is a more advanced stage. Unrecognised and null stages rank as UNKNOWN.
    """
    return (
        stage
        .replace(STAGE_FOR_MAX_MAPPING)
        .replace_strict(CATEGORY_RANKS_STR, default=_DEFAULT_STAGE_RANK_VALUE, return_dtype=pl.Int64)
    )


def _stage_name_from_rank(rank: pl.Expr) -> pl.Expr:
    """Map a stage rank back to its display name."""
    return rank.replace_strict(RANK_TO_CATEGORY_STR, default=_DEFAULT_STAGE_NAME_VALUE, return_dtype=pl.String)


def _friendly_stage_label(stage: str | None) -> str | None:
    """Turn a stage code such as PHASE_1_2 into the prose form Phase 1 2."""
    if stage is None:
        return None
    return stage.replace('_', ' ').lower().title()


def drug_molecule(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate the drug molecule index.

    Args:
        source: Dictionary with paths to:
            - molecule: Processed molecule parquet
            - chemical_probes: Chemical probes parquet
            - mechanism_of_action: Mechanism of action parquet
            - clinical_report: Clinical report parquet from clinical_report step
            - disease: Disease/EFO parquet
        destination: Dictionary with paths to:
            - output: Path to write the output parquet file.
            - excluded: Path to write excluded clinical reports that failed QC.
        settings: Custom settings with:
            - invalid_clinical_report_qc: List of QC reason strings to exclude.
        config: Config object (not used in this transformer).
    """
    logger.info(f'Loading data from {source}')
    clinical_report = scan_dataset(source['clinical_report']).collect()

    invalid_qc_reasons = settings.get('invalid_clinical_report_qc', [])
    if invalid_qc_reasons and 'qualityControls' in clinical_report.columns:
        has_invalid_qc = (
            pl.col('qualityControls').fill_null([]).list.set_intersection(invalid_qc_reasons).list.len() > 0
        )
        excluded = clinical_report.filter(has_invalid_qc)
        clinical_report = clinical_report.filter(~has_invalid_qc)
    else:
        excluded = clinical_report.clear()

    logger.info(f'Writing {excluded.height} excluded clinical reports to {destination["excluded"]}')
    write_dataset(excluded, str(destination['excluded']))

    molecule = scan_dataset(source['molecule'])
    chemical_probes = scan_dataset(source['chemical_probes']).collect()
    mechanism_of_action = scan_dataset(source['mechanism_of_action']).collect()
    disease = scan_dataset(source['disease']).collect()

    logger.info('Processing drug index')
    output = process_drug_index(molecule, chemical_probes, mechanism_of_action, clinical_report, disease)
    logger.info(f'Drug index has {output.height} molecules')

    logger.info(f'Writing drug index to {destination["output"]}')
    write_dataset(output, str(destination['output']))


def process_drug_index(
    molecule: pl.LazyFrame,
    chemical_probes: pl.DataFrame,
    mechanism_of_action: pl.DataFrame,
    clinical_report: pl.DataFrame,
    disease: pl.DataFrame,
) -> pl.DataFrame:
    """Process and combine all drug data into the final index.

    Args:
        molecule: Processed molecule data.
        chemical_probes: Chemical probes data.
        mechanism_of_action: Mechanism of action data.
        clinical_report: Clinical report data with drugs, diseases, and clinicalStage.
        disease: Disease/EFO data for indication mapping.

    Returns:
        Final drug index DataFrame.
    """
    max_phase = _compute_max_phase_per_drug(clinical_report)
    indications = _process_clinical_report_indications(clinical_report, disease)

    # Every chemical probe drug id, for the is_drug filter.
    probe_drug_ids = (
        chemical_probes
        .select(pl.col('drugId').alias('id'))
        .drop_nulls()
        .unique()
        .with_columns(_isChemicalProbe=pl.lit(value=True))
    )

    # Probe compound ids grouped per drug, for the cross-references.
    probe_xrefs = (
        chemical_probes
        .filter(pl.col('drugId').is_not_null())
        .sort('drugFromSourceId')
        .group_by(pl.col('drugId').alias('id'), maintain_order=True)
        .agg(pl.col('drugFromSourceId').drop_nulls().unique(maintain_order=True).alias('_probeIds'))
    )

    has_mechanism = (
        mechanism_of_action
        .select(pl.col('chemblIds').alias('id'))
        .explode('id')
        # exploding a null or empty list yields a null row, which is not a mechanism
        .drop_nulls()
        .unique()
        .with_columns(_hasMechanismOfAction=pl.lit(value=True))
    )

    drug = (
        molecule
        .join(max_phase.lazy(), on='id', how='left')
        .join(indications.lazy(), on='id', how='left')
        .join(probe_drug_ids.lazy(), on='id', how='left')
        .join(probe_xrefs.lazy(), on='id', how='left')
        .join(has_mechanism.lazy(), on='id', how='left')
        .with_columns(crossReferences=_with_probe_xref())
        .filter(_is_drug())
        .collect()
    )

    return (
        drug
        # the description reads maximumClinicalStage before it is defaulted below, so a
        # molecule with no clinical report gets no clinical stage clause at all
        .with_columns(description=_describe())
        .with_columns(maximumClinicalStage=pl.col('maximumClinicalStage').fill_null(_DEFAULT_STAGE_NAME_VALUE))
        # molecule ids are expected to be unique; this guards against an upstream regression
        .unique(subset=['id'], keep='first', maintain_order=True)
        .sort('id')
        .select(drug_molecule_schema.keys())
        .cast(drug_molecule_schema)
    )


def _with_probe_xref() -> pl.Expr:
    """Append the Probes&Drugs cross-reference when the molecule is a chemical probe.

    Most molecules have no cross-references at all, and `pl.concat_list` returns null
    rather than a one-element list when the list it is given is null, so that case is
    handled on its own branch.
    """
    probe_xref = pl.struct(source=pl.lit(PROBES_AND_DRUGS_SOURCE), ids=pl.col('_probeIds'))
    return (
        pl
        .when(pl.col('_probeIds').is_null())
        .then(pl.col('crossReferences'))
        .when(pl.col('crossReferences').is_null())
        .then(pl.concat_list(probe_xref))
        .otherwise(pl.concat_list('crossReferences', probe_xref))
    )


def _is_drug() -> pl.Expr:
    """Whether a molecule qualifies as a drug.

    True when it has a drugbank cross-reference, appears in a clinical report, has a
    mechanism of action, or is a chemical probe. A molecule that fails all four and has
    no cross-references at all yields null rather than false, which `filter` also drops.
    """
    return (
        pl.col('crossReferences').list.eval(pl.element().struct.field('source')).list.contains(DRUGBANK_SOURCE)
        | pl.col('maximumClinicalStage').is_not_null()
        | pl.col('_hasMechanismOfAction').is_not_null()
        | pl.col('_isChemicalProbe').is_not_null()
    )


def _describe() -> pl.Expr:
    """Build the human-readable description from the drug type, stage and indications."""
    return (
        pl
        .struct('drugType', 'maximumClinicalStage', 'indications')
        .map_elements(
            lambda row: _generate_description(
                row['drugType'],
                row['maximumClinicalStage'],
                [i['maxClinicalStage'] for i in row['indications'] or []],
                [i['efoName'] for i in row['indications'] or []],
            ),
            return_dtype=pl.String,
        )
    )


def _compute_max_phase_per_drug(clinical_report: pl.DataFrame) -> pl.DataFrame:
    """Compute the overall maximum clinical stage for each drug across all clinical reports.

    Explodes the drugs array, maps WITHDRAWAL/PHASE_4 to APPROVAL, ranks stages,
    and returns the best (most advanced) stage per drug.

    Args:
        clinical_report: Clinical report DataFrame with drugs array and clinicalStage.

    Returns:
        DataFrame with columns: id (drugId), maximumClinicalStage (string display name).
    """
    return (
        clinical_report
        .select('drugs', 'clinicalStage')
        .explode('drugs')
        .select(
            pl.col('drugs').struct.field('drugId').alias('id'),
            'clinicalStage',
        )
        .filter(pl.col('id').is_not_null())
        .group_by('id')
        .agg(_stage_rank(pl.col('clinicalStage')).min().alias('bestRank'))
        .select('id', maximumClinicalStage=_stage_name_from_rank(pl.col('bestRank')))
    )


def _process_clinical_report_indications(
    clinical_report: pl.DataFrame,
    disease: pl.DataFrame,
) -> pl.DataFrame:
    """Process clinical reports to extract per-drug, per-indication max stage.

    Explodes both drugs and diseases arrays, computes the best clinical stage
    per (drugId, diseaseId) pair, joins with disease data for names, and
    aggregates into an array of indication structs per drug.

    Args:
        clinical_report: Clinical report DataFrame with drugs, diseases, clinicalStage.
        disease: Disease/EFO DataFrame with id and name columns.

    Returns:
        DataFrame with columns: id (drugId), indications (array of structs).
    """
    exploded = (
        clinical_report
        .select('drugs', 'diseases', 'clinicalStage')
        .explode('drugs')
        .explode('diseases')
        .select(
            pl.col('drugs').struct.field('drugId'),
            pl.col('diseases').struct.field('diseaseId'),
            'clinicalStage',
        )
        .filter(pl.col('drugId').is_not_null() & pl.col('diseaseId').is_not_null())
    )

    per_indication = (
        exploded
        .group_by('drugId', 'diseaseId')
        .agg(_stage_rank(pl.col('clinicalStage')).min().alias('bestRank'))
        .with_columns(maxClinicalStage=_stage_name_from_rank(pl.col('bestRank')))
    )

    # only the space character is stripped from the name: a bare `strip_chars()` would
    # also take the other unicode whitespace a handful of disease names carry, changing
    # the name rather than trimming it
    disease_names = disease.select(
        pl.col('id').alias('diseaseId'),
        pl.col('name').str.to_lowercase().str.strip_chars(' ').alias('efoName'),
    )

    return (
        per_indication
        .join(disease_names, on='diseaseId', how='left')
        .sort(_INDICATION_SORT)
        .group_by(pl.col('drugId').alias('id'), maintain_order=True)
        .agg(
            pl
            .struct(
                pl.col('diseaseId').alias('disease'),
                pl.col('efoName'),
                pl.col('maxClinicalStage'),
            )
            .alias('indications')
        )
    )


def _generate_description(
    drug_type: str | None,
    max_phase: str | None,
    indication_stages: list[str | None] | None,
    indication_labels: list[str | None] | None,
) -> str:
    """Generate a human-readable description of a drug.

    Args:
        drug_type: Type of drug (e.g., "Small molecule").
        max_phase: Maximum clinical stage as a display name (e.g., "approved").
        indication_stages: List of per-indication max clinical stage display names.
        indication_labels: List of indication disease names.

    Returns:
        Human-readable description string.
    """
    if drug_type is None:
        drug_type = 'Unknown'

    main_note = f'{drug_type.capitalize()} drug'

    phase_str = ''
    if max_phase is not None:
        label_count = len(indication_labels) if indication_labels else 0
        multi_indication = ' (across all indications)' if label_count > 1 else ''
        phase_label = _friendly_stage_label(max_phase) or max_phase
        phase_str = f' with a maximum clinical stage of {phase_label}{multi_indication}'

    indication_str = ''
    if indication_stages is not None and indication_labels is not None:
        pairs = zip(indication_stages, indication_labels, strict=False)
        # dict.fromkeys deduplicates without disturbing the order
        indications = list(dict.fromkeys((s, lbl) for s, lbl in pairs if s is not None and lbl is not None))

        # sorted, because with two or fewer approved indications the labels are named in
        # the sentence and their order is visible to the reader
        approved = sorted(label for stage, label in indications if stage == APPROVED_STAGE_CODE)
        investigational_count = sum(1 for stage, _ in indications if stage != APPROVED_STAGE_CODE)

        if approved and not investigational_count:
            if len(approved) <= 2:
                indication_str = f', with an approval for {_join_semantic(approved)}'
            else:
                indication_str = f', with an approval for {len(approved)} indications'
        elif not approved and investigational_count:
            s = 's' if investigational_count > 1 else ''
            indication_str = f', with {investigational_count} investigational indication{s}'
        elif approved and investigational_count:
            s = 's' if investigational_count > 1 else ''
            if len(approved) <= 2:
                approved_str = _join_semantic(approved)
                indication_str = (
                    f', with an approval for {approved_str} and {investigational_count} investigational indication{s}'
                )
            else:
                indication_str = (
                    f', with {len(approved)} approved and {investigational_count} investigational indication{s}'
                )

    return f'{main_note}{phase_str}{indication_str}.'


def _join_semantic(items: list[str]) -> str:
    """Join items in a grammatically correct way.

    Args:
        items: List of strings to join.

    Returns:
        Joined string (e.g., "a, b and c").
    """
    if not items:
        return ''
    if len(items) == 1:
        return items[0]
    return f'{", ".join(items[:-1])} and {items[-1]}'
