"""Candidate mechanisms of action from the staged AACT extraction.

Links each pooled drug name to a ChEMBL parent through the trials of its pool, grounds the
extracted targets on Ensembl genes, and keeps the direct, confident mechanisms.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import polars as pl
from loguru import logger

from pts.transformers.clinical_mira.moa.benchmark import link_pools
from pts.transformers.clinical_mira.moa.grounding import GROUNDED, build_label_index, ground_mechanisms
from pts.transformers.clinical_mira.staged_outputs import MOA
from pts.transformers.drug_molecule import compute_max_phase_per_drug
from pts.transformers.utils.dataset import scan_dataset, write_dataset

if TYPE_CHECKING:
    from otter.config.model import Config

LABEL_COLUMNS = ['id', 'approvedSymbol', 'symbolSynonyms', 'obsoleteSymbols', 'approvedName', 'nameSynonyms']

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
    'nctIds',
    'quote',
    'quoteNctId',
]


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


def build_candidates(
    moa: pl.DataFrame,
    clinical_report: pl.DataFrame,
    chembl_molecule: pl.DataFrame,
    target: pl.DataFrame,
    *,
    link_share: float,
    class_level_genes: int,
    report_min_confidence: float,
) -> pl.DataFrame:
    """Link, ground and filter the staged mechanisms.

    Args:
        moa: The staged snapshot, one row per mechanism.
        clinical_report: The clinical report, which links pools and gives the stages.
        chembl_molecule: The ChEMBL molecule intermediate.
        target: The target dataset, whose labels ground the target names.
        link_share: The share of a pool's single-drug trials one drug must cover to link it.
        class_level_genes: A target grounding to more genes than this names a class.
        report_min_confidence: The lowest confidence a mechanism may have.

    Returns:
        pl.DataFrame: One row per surviving mechanism, with ``CANDIDATE_COLUMNS``.
    """
    pools = (
        moa
        .group_by('drugName', maintain_order=True)
        .agg(pl.col('nctIds').first())
        .rename({'drugName': 'pool_id', 'nctIds': 'nct_ids'})
    )
    molecules = _linking_molecules(chembl_molecule, clinical_report)
    links = (
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
        .drop_nulls('parentId')
        .select(
            drugName='pool_id',
            parentChemblId='parentId',
            maximumClinicalStage='maximumClinicalStage',
            drugType='drugType',
        )
    )
    family = chembl_molecule.select(
        parentChemblId='id',
        chemblIds=pl.concat_list(pl.col('id'), pl.col('childChemblIds').fill_null([])).list.unique(maintain_order=True),
    )

    kept = moa.filter(
        (pl.col('relation') == 'direct')
        & (pl.col('statementType') != 'hypothesis')
        & (pl.col('actionType') != 'OTHER')
        & pl.col('isHumanTarget')
        & (pl.col('confidence') >= report_min_confidence)
    )
    index = build_label_index(target.select(LABEL_COLUMNS))
    grounded = ground_mechanisms(kept.with_columns(kept=pl.lit(True)), index, class_level_genes).filter(
        pl.col('grounding').is_in(GROUNDED) & ~pl.col('class_level')
    )
    cited = pl.col('evidence').list.eval(pl.element().struct.field('nctId').str.to_uppercase())
    return (
        grounded
        .join(links, on='drugName', how='inner', maintain_order='left')
        .join(family, on='parentChemblId', how='left', maintain_order='left')
        .select(
            'parentChemblId',
            'chemblIds',
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
            nctIds=cited.list.unique().list.sort(),
            quote=pl.col('evidence').list.first().struct.field('quote'),
            quoteNctId=cited.list.first(),
        )
        .select(CANDIDATE_COLUMNS)
    )


def drug_mechanism_of_action_aact(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Write the candidate mechanisms of action extracted from the AACT trial texts.

    Args:
        source: Paths to ``moa`` (the staged snapshot), ``clinical_report``, ``chembl_molecule`` and ``target``.
        destination: Path to ``candidates``.
        settings: ``link_share``, ``class_level_genes`` and ``report_min_confidence``.
        config: Config object (not used).
    """
    logger.info(f'Reading the staged mechanisms from {source["moa"]}')
    moa = MOA.read(source['moa'])
    clinical_report = scan_dataset(source['clinical_report']).select('id', 'source', 'drugs', 'clinicalStage').collect()
    chembl_molecule = (
        scan_dataset(source['chembl_molecule']).select('id', 'parentId', 'childChemblIds', 'drugType').collect()
    )
    target = scan_dataset(source['target']).select(LABEL_COLUMNS).collect()

    candidates = build_candidates(
        moa,
        clinical_report,
        chembl_molecule,
        target,
        link_share=settings['link_share'],
        class_level_genes=settings['class_level_genes'],
        report_min_confidence=settings['report_min_confidence'],
    )
    logger.info(f'Writing {candidates.height} candidates of {moa.height} mechanisms')
    write_dataset(candidates, destination['candidates'])
