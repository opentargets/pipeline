"""Candidate mechanisms of action from the staged trial and literature extractions.

Trial mechanisms are linked to a ChEMBL parent through the trials of their pooled drug name;
literature mechanisms already name a ChEMBL drug, re-parented here. Both are grounded on Ensembl
genes and filtered to the direct, confident mechanisms.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.transformers.clinical_mira.moa.benchmark import link_pools
from pts.transformers.clinical_mira.moa.grounding import (
    FIELDS,
    GROUNDED,
    LOCATION_FIELDS,
    build_label_index,
    ground_mechanisms,
)
from pts.transformers.drug_molecule import compute_max_phase_per_drug

LABEL_COLUMNS = ['id', *FIELDS, *LOCATION_FIELDS]

SOURCE_AACT = 'aact'
SOURCE_LITERATURE = 'literature'

CANDIDATE_COLUMNS = [
    'parentChemblId',
    'chemblIds',
    'maximumClinicalStage',
    'drugType',
    'targetFromSource',
    'targets',
    'targetSymbols',
    'targetType',
    'actionType',
    'mechanismOfAction',
    'confidence',
    'statementType',
    'source',
    'docIds',
    'quote',
    'quoteDocId',
]


def stage_excluded(candidates: pl.DataFrame, exclude_stages: dict[str, list[str]]) -> pl.Expr:
    """Whether a candidate's maximum clinical stage is one its source excludes.

    Args:
        candidates: The candidates the expression is for, whose ``source`` values are checked.
        exclude_stages: For each source, the maximum clinical stages excluded.

    Returns:
        pl.Expr: True for a candidate whose source excludes its stage, False otherwise.

    Raises:
        ValueError: If a source of ``candidates`` has no ``exclude_stages`` entry.
    """
    missing = sorted(set(candidates['source'].unique().to_list()) - set(exclude_stages))
    if missing:
        msg = f'exclude_stages has no entry for the candidate sources {missing}'
        raise ValueError(msg)
    excluded = pl.lit(False)
    for source, stages in exclude_stages.items():
        excluded = excluded | (
            (pl.col('source') == source)
            & pl.col('maximumClinicalStage').is_in(pl.lit(stages, dtype=pl.List(pl.String))).fill_null(False)
        )
    return excluded


def _linking_molecules(chembl_molecule: pl.DataFrame, clinical_report: pl.DataFrame) -> pl.DataFrame:
    """Each ChEMBL molecule with the stage and drug type of its parent.

    The stage is the parent's maximum clinical stage in the clinical report, or the molecule's own
    when the parent has none.

    Returns:
        pl.DataFrame: ``id``, ``parentId``, ``maximumClinicalStage``, ``drugType``.
    """
    molecules = chembl_molecule.select('id', parentId=pl.coalesce('parentId', 'id'), drugType='drugType')
    stages = compute_max_phase_per_drug(clinical_report)
    parent_types = molecules.select(parentId='id', parentType='drugType').unique('parentId')
    return (
        molecules
        .join(parent_types, on='parentId', how='left', maintain_order='left')
        .join(stages.rename({'maximumClinicalStage': 'ownStage'}), on='id', how='left', maintain_order='left')
        .join(
            stages.rename({'id': 'parentId', 'maximumClinicalStage': 'parentStage'}),
            on='parentId',
            how='left',
            maintain_order='left',
        )
        .select(
            'id',
            'parentId',
            maximumClinicalStage=pl.coalesce('parentStage', 'ownStage'),
            drugType=pl.coalesce('parentType', 'drugType'),
        )
    )


def _grounded(
    moa: pl.DataFrame,
    target: pl.DataFrame,
    *,
    class_level_genes: int,
    report_min_confidence: float,
    suggestion_min_confidence: float,
) -> pl.DataFrame:
    """The direct, confident, human mechanisms that ground to a specific set of genes.

    Returns:
        pl.DataFrame: ``moa``'s surviving rows, with ``targetIds`` and ``targetSymbols``.
    """
    kept = moa.filter(
        (pl.col('relation') == 'direct')
        & (pl.col('statementType') != 'hypothesis')
        & (pl.col('actionType') != 'OTHER')
        & pl.col('isHumanTarget')
        & (pl.col('confidence') >= report_min_confidence)
    )
    index = build_label_index(target.select(LABEL_COLUMNS))
    return ground_mechanisms(
        kept.with_columns(kept=pl.lit(True)), index, class_level_genes, suggestion_min_confidence
    ).filter(pl.col('grounding').is_in(GROUNDED) & ~pl.col('class_level'))


def _with_family(frame: pl.DataFrame, chembl_molecule: pl.DataFrame) -> pl.DataFrame:
    """Add ``chemblIds``, the parent and its child molecules, to rows keyed by ``parentChemblId``."""
    family = chembl_molecule.select(
        parentChemblId='id',
        chemblIds=pl.concat_list(pl.col('id'), pl.col('childChemblIds').fill_null([])).list.unique(maintain_order=True),
    )
    return frame.join(family, on='parentChemblId', how='left', maintain_order='left')


def _candidates(grounded: pl.DataFrame, source: str, cited: pl.Expr) -> pl.DataFrame:
    """Select ``CANDIDATE_COLUMNS`` from linked, grounded mechanisms citing the ``cited`` doc ids."""
    return grounded.select(
        'parentChemblId',
        pl.coalesce('chemblIds', pl.concat_list('parentChemblId')).alias('chemblIds'),
        'maximumClinicalStage',
        'drugType',
        'targetFromSource',
        targets='targetIds',
        targetSymbols='targetSymbols',
        targetType=pl.col('targetType').str.to_lowercase(),
        actionType='actionType',
        mechanismOfAction='mechanismOfAction',
        confidence='confidence',
        statementType='statementType',
        source=pl.lit(source),
        docIds=cited.list.unique().list.sort(),
        quote=pl.col('evidence').list.first().struct.field('quote'),
        quoteDocId=cited.list.first(),
    ).select(CANDIDATE_COLUMNS)


def build_trial_candidates(
    moa: pl.DataFrame,
    clinical_report: pl.DataFrame,
    chembl_molecule: pl.DataFrame,
    target: pl.DataFrame,
    *,
    link_share: float,
    class_level_genes: int,
    report_min_confidence: float,
    suggestion_min_confidence: float,
) -> pl.DataFrame:
    """Link, ground and filter the staged trial mechanisms.

    Args:
        moa: The staged trial snapshot, one row per mechanism.
        clinical_report: The clinical report, which links pools and gives the stages.
        chembl_molecule: The ChEMBL molecule intermediate.
        target: The target dataset, whose labels ground the target names.
        link_share: The share of a pool's single-drug trials one drug must cover to link it.
        class_level_genes: A target grounding to more genes than this names a class.
        report_min_confidence: The lowest confidence a mechanism may have.
        suggestion_min_confidence: The lowest confidence at which a mechanism whose target name
            does not support the model's symbol suggestion still grounds to that suggestion.

    Returns:
        pl.DataFrame: One row per surviving mechanism, with ``CANDIDATE_COLUMNS``, ``source``
        ``aact`` and the cited NCT ids, upper-cased, as ``docIds``.
    """
    pools = (
        moa
        .group_by('drugName', maintain_order=True)
        .agg(pl.col('nctIds').first())
        .rename({'drugName': 'pool_id', 'nctIds': 'nct_ids'})
    )
    molecules = _linking_molecules(chembl_molecule, clinical_report)
    linked = (
        link_pools(
            pools,
            clinical_report,
            molecules.select('id', 'maximumClinicalStage', 'drugType'),
            link_share,
        )
        .filter(pl.col('link') == 'linked')
        .join(
            molecules.select(pl.col('id').alias('drugId'), 'parentId'), on='drugId', how='left', maintain_order='left'
        )
    )
    absent = linked.filter(pl.col('parentId').is_null()).height
    if absent:
        logger.warning(f'{absent} linked pools dropped: their linked drug is absent from chembl_molecule')
    links = linked.drop_nulls('parentId').select(
        drugName='pool_id',
        parentChemblId='parentId',
        maximumClinicalStage='maximumClinicalStage',
        drugType='drugType',
    )
    grounded = _grounded(
        moa,
        target,
        class_level_genes=class_level_genes,
        report_min_confidence=report_min_confidence,
        suggestion_min_confidence=suggestion_min_confidence,
    ).join(links, on='drugName', how='inner', maintain_order='left')
    cited = pl.col('evidence').list.eval(pl.element().struct.field('nctId').str.to_uppercase())
    return _candidates(_with_family(grounded, chembl_molecule), SOURCE_AACT, cited)


def build_literature_candidates(
    moa: pl.DataFrame,
    clinical_report: pl.DataFrame,
    chembl_molecule: pl.DataFrame,
    target: pl.DataFrame,
    *,
    class_level_genes: int,
    report_min_confidence: float,
    suggestion_min_confidence: float,
) -> pl.DataFrame:
    """Re-parent, ground and filter the staged literature mechanisms.

    Each ``drugId`` is re-parented through ``chembl_molecule``; mechanisms whose drug is absent
    from it are dropped and counted.

    Args:
        moa: The staged literature snapshot, one row per mechanism of a ``drugId``.
        clinical_report: The clinical report, which gives the stages.
        chembl_molecule: The ChEMBL molecule intermediate.
        target: The target dataset, whose labels ground the target names.
        class_level_genes: A target grounding to more genes than this names a class.
        report_min_confidence: The lowest confidence a mechanism may have.
        suggestion_min_confidence: The lowest confidence at which a mechanism whose target name
            does not support the model's symbol suggestion still grounds to that suggestion.

    Returns:
        pl.DataFrame: One row per surviving mechanism, with ``CANDIDATE_COLUMNS``, ``source``
        ``literature`` and the cited ``pmid…`` and ``pmc…`` doc ids as ``docIds``.
    """
    molecules = _linking_molecules(chembl_molecule, clinical_report).select(
        drugId='id',
        parentChemblId='parentId',
        maximumClinicalStage='maximumClinicalStage',
        drugType='drugType',
    )
    resolved = moa.join(molecules, on='drugId', how='left', maintain_order='left')
    absent = resolved.filter(pl.col('parentChemblId').is_null())
    if absent.height:
        logger.warning(
            f'{absent.height} literature mechanisms of {absent["drugId"].n_unique()} drugs dropped: '
            'their drug is absent from chembl_molecule'
        )
    grounded = _grounded(
        resolved.drop_nulls('parentChemblId'),
        target,
        class_level_genes=class_level_genes,
        report_min_confidence=report_min_confidence,
        suggestion_min_confidence=suggestion_min_confidence,
    )
    cited = pl.col('evidence').list.eval(pl.element().struct.field('nctId').str.to_lowercase())
    return _candidates(_with_family(grounded, chembl_molecule), SOURCE_LITERATURE, cited)
