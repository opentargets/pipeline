"""Clinical success of drug target-disease pairs with genetic support.

For each evidence datasource and clinical phase, tests whether the target-disease pairs drugs were
taken to the clinic for reach that phase more often when the datasource supports the pair, after
Minikel et al. 2024 (doi.org/10.1038/s41586-024-07316-0). A pair is supported when its indirect
association, the highest evidence score over the disease and its descendants, reaches
`score_threshold`. Ported from gentropy `chemblDrugEnrichment.drug_enrichemnt_from_evidence`.

Not part of the unified pipeline: run on demand against a release.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config
from scipy.stats import chi2, fisher_exact

from pts.schemas.dataset.conform import without_enums
from pts.schemas.drug_enrichment import drug_enrichment_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset

#: Clinical phase of each `clinicalStage`. Other stages are left out.
CLINICAL_STAGE_PHASES = {
    'EARLY_PHASE_1': 1,
    'PHASE_1': 1,
    'PHASE_1_2': 1,
    'PHASE_2': 2,
    'PHASE_2_3': 2,
    'PHASE_3': 3,
    'PREAPPROVAL': 3,
    'APPROVAL': 4,
    'PHASE_4': 4,
}

#: Each test compares the pairs that reached the phase with the pairs that did not.
PHASES = (2, 3, 4)

#: z of the 95% confidence intervals.
Z = 1.96

#: Source keys that are not evidence datasets to test.
_FIXED_SOURCES = {'disease', 'drug_mechanism_of_action', 'clinical_precedence'}


@dataclass(frozen=True)
class Settings:
    """The step's settings, with their defaults."""

    #: Lowest indirect association score that counts as genetic support.
    score_threshold: float = 0.5
    #: Diseases under these ancestors are left out of both sides (cancer or benign tumor).
    excluded_ancestors: tuple[str, ...] = ('MONDO_0045024',)
    #: Drugs with more distinct targets than this are left out.
    max_drug_targets: int = 5

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> Settings:
        """Read the settings from the step's config.

        Args:
            settings: any of the fields.

        Returns:
            The settings.

        Raises:
            ValueError: for an unknown setting.
        """
        unknown = set(settings) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown settings {sorted(unknown)}')
        return cls(
            score_threshold=float(settings.get('score_threshold', cls.score_threshold)),
            excluded_ancestors=tuple(settings.get('excluded_ancestors', cls.excluded_ancestors)),
            max_drug_targets=int(settings.get('max_drug_targets', cls.max_drug_targets)),
        )


def disease_lineage(disease: pl.LazyFrame) -> pl.LazyFrame:
    """Each disease with each of its ancestors, itself included.

    Args:
        disease: the disease dataset.

    Returns:
        `diseaseId` and `ancestorId`.
    """
    return (
        disease
        .select(
            pl.col('id').alias('diseaseId'),
            pl.concat_list(pl.col('id'), pl.col('ancestors').fill_null(pl.lit([], dtype=pl.List(pl.String)))).alias(
                'ancestorId'
            ),
        )
        .explode('ancestorId')
        .unique()
    )


def excluded_diseases(lineage: pl.LazyFrame, ancestors: tuple[str, ...]) -> pl.LazyFrame:
    """The diseases under any of `ancestors`, the ancestors included.

    Args:
        lineage: from `disease_lineage`.
        ancestors: disease ids.

    Returns:
        `diseaseId`.
    """
    return lineage.filter(pl.col('ancestorId').is_in(ancestors)).select('diseaseId').unique()


def promiscuous_drugs(drug_mechanism_of_action: pl.LazyFrame, max_targets: int) -> pl.LazyFrame:
    """The drugs with more than `max_targets` distinct targets over all their mechanisms.

    Args:
        drug_mechanism_of_action: the drug mechanism of action dataset.
        max_targets: the most targets a drug may have.

    Returns:
        `drugId`.
    """
    return (
        drug_mechanism_of_action
        .select(pl.col('chemblIds').alias('drugId'), pl.col('targets').alias('targetId'))
        .explode('drugId')
        .explode('targetId')
        .drop_nulls()
        .group_by('drugId')
        .agg(pl.col('targetId').n_unique().alias('targets'))
        .filter(pl.col('targets') > max_targets)
        .select('drugId')
    )


def clinical_pairs(
    clinical_precedence: pl.LazyFrame, excluded: pl.LazyFrame, promiscuous: pl.LazyFrame
) -> pl.LazyFrame:
    """The highest clinical phase of each target-disease pair.

    Args:
        clinical_precedence: the clinical precedence evidence.
        excluded: from `excluded_diseases`.
        promiscuous: from `promiscuous_drugs`.

    Returns:
        `targetId`, `diseaseId` and `maxPhase`.
    """
    return (
        without_enums(clinical_precedence)
        .select('targetId', 'diseaseId', 'drugId', 'clinicalStage')
        .join(promiscuous, on='drugId', how='anti')
        .join(excluded, on='diseaseId', how='anti')
        .with_columns(
            pl
            .col('clinicalStage')
            .replace_strict(CLINICAL_STAGE_PHASES, default=None, return_dtype=pl.Int64)
            .alias('phase')
        )
        .drop_nulls('phase')
        .group_by('targetId', 'diseaseId')
        .agg(pl.col('phase').max().alias('maxPhase'))
    )


def indirect_scores(evidence: pl.LazyFrame, lineage: pl.LazyFrame, excluded: pl.LazyFrame) -> pl.LazyFrame:
    """The indirect association score of each datasource, target and disease.

    Args:
        evidence: an evidence dataset.
        lineage: from `disease_lineage`.
        excluded: from `excluded_diseases`.

    Returns:
        `datasourceId`, `targetId`, `diseaseId` and `score`, the highest evidence score over the
        disease and its descendants.
    """
    return (
        without_enums(evidence)
        .select('datasourceId', 'targetId', 'diseaseId', 'score')
        .join(excluded, on='diseaseId', how='anti')
        .join(lineage, on='diseaseId')
        .group_by('datasourceId', 'targetId', 'ancestorId')
        .agg(pl.col('score').max())
        .rename({'ancestorId': 'diseaseId'})
    )


def phase_test(supported: int, supported_reached: int, unsupported: int, unsupported_reached: int) -> dict[str, float]:
    """Fisher's exact odds ratio and the relative success of reaching a phase with genetic support.

    With an empty cell, both ratios are 1 with no width and the relative success p-value is 1, as
    in gentropy.

    Args:
        supported: supported pairs.
        supported_reached: supported pairs that reached the phase.
        unsupported: unsupported pairs.
        unsupported_reached: unsupported pairs that reached the phase.

    Returns:
        The two ratios with their confidence intervals and p-values.
    """
    table = np.array([
        [unsupported - unsupported_reached, unsupported_reached],
        [supported - supported_reached, supported_reached],
    ])
    odds_ratio, p_value = fisher_exact(table)
    if (table == 0).any():
        odds_ratio, se_ln_or, relative_success, se_ln_rs, p_value_rs = 1.0, 0.0, 1.0, 0.0, 1.0
    else:
        se_ln_or = float(np.sqrt((1 / table).sum()))
        relative_success = (supported_reached / supported) / (unsupported_reached / unsupported)
        se_ln_rs = float(np.sqrt(1 / unsupported_reached - 1 / unsupported + 1 / supported_reached - 1 / supported))
        p_value_rs = float(chi2.sf((np.log(relative_success) / se_ln_rs) ** 2, df=1))
    ln_or, ln_rs = np.log(odds_ratio), np.log(relative_success)
    return {
        'oddsRatio': float(odds_ratio),
        'oddsRatioCiLow': float(np.exp(ln_or - Z * se_ln_or)),
        'oddsRatioCiHigh': float(np.exp(ln_or + Z * se_ln_or)),
        'oddsRatioPValue': float(p_value),
        'relativeSuccess': float(relative_success),
        'relativeSuccessCiLow': float(np.exp(ln_rs - Z * se_ln_rs)),
        'relativeSuccessCiHigh': float(np.exp(ln_rs + Z * se_ln_rs)),
        'relativeSuccessPValue': p_value_rs,
    }


def enrichment(clinical: pl.DataFrame, indirect: pl.DataFrame, score_threshold: float) -> list[dict[str, Any]]:
    """The phase tests of each datasource in `indirect`.

    Args:
        clinical: from `clinical_pairs`.
        indirect: from `indirect_scores`.
        score_threshold: lowest score that counts as genetic support.

    Returns:
        One row per datasource and phase, in `drug_enrichment_schema`.
    """
    rows = []
    for datasource_id in sorted(indirect['datasourceId'].unique()):
        support = indirect.filter(
            (pl.col('datasourceId') == datasource_id) & (pl.col('score') >= score_threshold)
        ).select('targetId', 'diseaseId', pl.lit(True).alias('supported'))
        pairs = clinical.join(support, on=['targetId', 'diseaseId'], how='left').with_columns(
            pl.col('supported').fill_null(False)
        )
        supported = pairs['supported']
        for phase in PHASES:
            reached = pairs['maxPhase'] >= phase
            counts = {
                'unsupportedBelowPhase': int((~supported & ~reached).sum()),
                'unsupportedReachedPhase': int((~supported & reached).sum()),
                'supportedBelowPhase': int((supported & ~reached).sum()),
                'supportedReachedPhase': int((supported & reached).sum()),
            }
            stats = phase_test(
                counts['supportedBelowPhase'] + counts['supportedReachedPhase'],
                counts['supportedReachedPhase'],
                counts['unsupportedBelowPhase'] + counts['unsupportedReachedPhase'],
                counts['unsupportedReachedPhase'],
            )
            rows.append({
                'datasourceId': datasource_id,
                'clinicalPhase': f'{phase}+',
                **stats,
                **counts,
                'supportedAssociations': support.height,
            })
    return rows


def drug_enrichment(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Write the drug enrichment table of each evidence dataset.

    Args:
        source: paths keyed by disease, drug_mechanism_of_action and clinical_precedence, and one
            `evidence_<name>` key per evidence dataset to test.
        destination: path of the drug enrichment table.
        settings: as `Settings.from_settings` reads them.
        config: unused.

    Raises:
        ValueError: when no evidence dataset is given.
    """
    options = Settings.from_settings(settings)
    evidence_keys = sorted(key for key in source if key not in _FIXED_SOURCES)
    if not evidence_keys:
        raise ValueError('no evidence dataset to test: add an evidence_<name> source')

    lineage = disease_lineage(scan_dataset(source['disease'])).collect()
    excluded = excluded_diseases(lineage.lazy(), options.excluded_ancestors).collect()
    promiscuous = promiscuous_drugs(
        scan_dataset(source['drug_mechanism_of_action']), options.max_drug_targets
    ).collect()
    clinical = clinical_pairs(
        scan_dataset(source['clinical_precedence']), excluded.lazy(), promiscuous.lazy()
    ).collect()
    logger.info(
        f'{excluded.height} excluded diseases, {promiscuous.height} drugs with more than '
        f'{options.max_drug_targets} targets, {clinical.height} clinical target-disease pairs'
    )

    rows = []
    for key in evidence_keys:
        indirect = indirect_scores(scan_dataset(source[key]), lineage.lazy(), excluded.lazy()).collect()
        rows.extend(enrichment(clinical, indirect, options.score_threshold))
    output = pl.DataFrame(rows, schema=drug_enrichment_schema)
    write_dataset(output, destination)
    logger.info(f'wrote {output.height} rows')
