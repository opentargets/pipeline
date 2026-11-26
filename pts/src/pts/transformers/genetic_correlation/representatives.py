"""One representative study per disease and LD ancestry.

Most pairs of GWAS studies are two cohorts of the same disease, so genetic correlation is computed
between diseases, each stood for by one study. Eligible are GWAS with summary statistics that pass
QC, are of a kind LDSC applies to, draw nearly all their samples from one LD ancestry with a score
panel, have a genome-wide significant variant, and whose own heritability is estimated with the
power stratified LDSC needs. Of those, the study with the most genome-wide significant credible
sets is chosen, then the one with the highest heritability z-score.

The effective sample size only screens out small studies: without case and control counts it is
the total sample size, which overstates the power of a biobank case-control study many times over.

Optionally, studies the release excluded only for failing the PZ or mean beta checks are a
fallback, chosen only for a disease and ancestry with no eligible release study. The release has
no credible sets of excluded studies, so their count is unknown, null, and they are chosen by
heritability z-score.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.genetic_correlation.previous import previous_path
from pts.transformers.genetic_correlation.settings import (
    LD_ANCESTRIES,
    RepresentativesDestination,
    RepresentativesSettings,
    RepresentativesSource,
)
from pts.transformers.utils.dataset import check_destinations_absent, scan_dataset, write_dataset

OUTPUT_SCHEMA = pl.Schema({
    'diseaseId': pl.String,
    'ldAncestry': pl.String,
    'studyId': pl.String,
    'effectiveSampleSize': pl.Float64,
    'nSamples': pl.Int32,
    'nCases': pl.Int32,
    'nControls': pl.Int32,
    'ancestryShare': pl.Float64,
    'nSignificantVariants': pl.Int64,
    'nSignificantCredibleSets': pl.Int64,
    'h2': pl.Float64,
    'h2StandardError': pl.Float64,
    'h2ZScore': pl.Float64,
    'lambdaGc': pl.Float64,
    'excludedFromRelease': pl.Boolean,
})

SIGNIFICANT_VARIANTS_CHECK = 'n_variants_sig'
"""The `sumstatQCValues` check counting the genome-wide significant variants of the summary statistics."""


CHANGES_SCHEMA = pl.Schema({
    'diseaseId': pl.String,
    'ldAncestry': pl.String,
    'studyId': pl.String,
    'previousStudyId': pl.String,
    'change': pl.String,
})
"""How each disease and ancestry's representative changed since the previous run: `added`,
`unchanged`, `replaced` or `dropped`. `studyId` is null for a dropped one, `previousStudyId` for an
added one, and every one is `added` without a previous run."""


def effective_sample_size() -> pl.Expr:
    """`4 * cases * controls / (cases + controls)` for a case-control study, else the sample size."""
    cases = pl.col('nCases').cast(pl.Float64)
    controls = pl.col('nControls').cast(pl.Float64)
    samples = pl.col('nSamples').cast(pl.Float64)
    return (
        pl
        .when((cases > 0) & (controls > 0))
        .then(4 * cases * controls / (cases + controls))
        .when(samples > 0)
        .then(samples)
        .otherwise(None)
        .alias('effectiveSampleSize')
    )


def significant_variants() -> pl.Expr:
    """The study's `n_variants_sig` QC value, null when it was not computed."""
    check = pl.element().struct.field('QCCheckName') == SIGNIFICANT_VARIANTS_CHECK
    return (
        pl
        .col('sumstatQCValues')
        .list.eval(pl.element().filter(check).struct.field('QCCheckValue'))
        .list.first()
        .cast(pl.Int64)
        .alias('nSignificantVariants')
    )


def ld_ancestry(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Each study's dominant LD population among those with a panel: `studyId`, `ldAncestry`, `ancestryShare`.

    Populations are weighted by `relativeSampleSize`; a tie goes to `nfe`, then alphabetically.
    `ancestryShare` is the population's share of the weight of every population, with a panel or not.
    A study with no population among `LD_ANCESTRIES` has no row.
    """
    entry = pl.col('ldPopulationStructure')
    total = pl.col('weight').sum().over('studyId')
    return (
        studies
        .select('studyId', 'ldPopulationStructure')
        .explode('ldPopulationStructure')
        .select(
            'studyId',
            entry.struct.field('ldPopulation').str.strip_chars().str.to_lowercase().alias('population'),
            entry.struct.field('relativeSampleSize').cast(pl.Float64).alias('weight'),
        )
        .filter(pl.col('population').is_not_null(), pl.col('weight').is_not_null())
        .group_by('studyId', 'population')
        .agg(pl.col('weight').sum())
        .with_columns(pl.when(total > 0).then(pl.col('weight') / total).alias('ancestryShare'))
        .filter(pl.col('population').is_in(LD_ANCESTRIES))
        .sort('studyId', 'weight', pl.col('population') == 'nfe', 'population', descending=[False, True, True, False])
        .unique('studyId', keep='first', maintain_order=True)
        .select('studyId', pl.col('population').alias('ldAncestry'), 'ancestryShare')
    )


def eligible_studies(
    studies: pl.LazyFrame, options: RepresentativesSettings, *, excluded: bool = False
) -> pl.LazyFrame:
    """The studies a representative may be chosen from, with their ancestry and effective sample size.

    Args:
        studies: a study index.
        options: the eligibility thresholds.
        excluded: `studies` are those the release excluded, which may fail the
            `excluded_tolerated_quality_controls` checks, and are marked `excludedFromRelease`.
    """
    disqualifying = [
        flag
        for flag in options.disqualifying_quality_controls
        if not (excluded and flag in options.excluded_tolerated_quality_controls)
    ]
    flags = pl.col('analysisFlags').list.eval(pl.element().str.strip_chars().str.to_lowercase())
    compatible = pl.col('analysisFlags').is_null() | (
        flags.list.set_difference(options.allowed_analysis_flags).list.len() == 0
    )
    disqualified = pl.col('qualityControls').fill_null([]).list.set_intersection(disqualifying).list.len() > 0
    return (
        studies
        .filter(pl.col('studyType') == 'gwas', pl.col('hasSumstats'), ~disqualified, compatible)
        .with_columns(effective_sample_size(), significant_variants())
        .filter(
            pl.col('effectiveSampleSize') >= options.min_effective_sample_size,
            pl.col('nSignificantVariants').fill_null(0) >= options.min_significant_variants,
        )
        .join(ld_ancestry(studies), on='studyId')
        .filter(pl.col('ldAncestry').is_in(options.ancestries), pl.col('ancestryShare') >= options.min_ancestry_share)
        .select(
            'studyId',
            'ldAncestry',
            'effectiveSampleSize',
            pl.col('nSamples').cast(pl.Int32),
            pl.col('nCases').cast(pl.Int32),
            pl.col('nControls').cast(pl.Int32),
            'ancestryShare',
            'nSignificantVariants',
            pl.lit(excluded).alias('excludedFromRelease'),
            pl.col(options.disease_column).alias('diseaseIds'),
        )
    )


def significant_loci(credible_sets: pl.LazyFrame, max_p_value: float) -> pl.LazyFrame:
    """Each study's number of credible sets whose lead variant has a p-value of at most `max_p_value`.

    The p-value is compared in log10 space with a relative tolerance, so a lead at exactly
    `max_p_value`, written as a mantissa and exponent, counts.
    """
    log_p = pl.col('pValueMantissa').cast(pl.Float64).log10() + pl.col('pValueExponent').cast(pl.Float64)
    threshold = math.log10(max_p_value)
    return (
        credible_sets
        .filter(log_p <= threshold + 1e-9 * abs(threshold))
        .group_by('studyId')
        .agg(pl.len().cast(pl.Int64).alias('nSignificantCredibleSets'))
    )


def successful_heritability(heritability: pl.LazyFrame) -> pl.LazyFrame:
    """One successful estimate per study, chosen as the `study` step chooses it."""
    fields = ['h2', 'h2_se', 'lambda_gc']
    return (
        heritability
        .filter(pl.col('runStatus') == 'success')
        .select('studyId', *fields)
        .sort(['studyId', *fields], nulls_last=True)
        .unique('studyId', keep='first', maintain_order=True)
        .select(
            'studyId',
            pl.col('h2').cast(pl.Float64),
            pl.col('h2_se').cast(pl.Float64).alias('h2StandardError'),
            pl.col('lambda_gc').cast(pl.Float64).alias('lambdaGc'),
        )
    )


def heritable_studies(
    studies: pl.LazyFrame, heritability: pl.LazyFrame, options: RepresentativesSettings
) -> pl.LazyFrame:
    """The eligible studies whose heritability estimate passes the thresholds, with `h2ZScore`."""
    error = pl.when(pl.col('h2StandardError') > 0).then(pl.col('h2StandardError'))
    lambda_ok = pl.col('lambdaGc').is_null() | pl.col('lambdaGc').is_between(
        options.min_lambda_gc, options.max_lambda_gc
    )
    return (
        studies
        .join(heritability, on='studyId')
        .with_columns((pl.col('h2') / error).alias('h2ZScore'))
        .filter(
            pl.col('h2') > options.min_h2,
            pl.col('h2') < options.max_h2,
            pl.col('h2ZScore') >= options.min_h2_z,
            lambda_ok,
        )
    )


def choose_representatives(candidates: pl.LazyFrame) -> pl.LazyFrame:
    """The best study per disease and ancestry.

    A release study before an excluded one, then the most significant credible sets, then the
    highest `h2ZScore`, then the first `studyId`.
    """
    return (
        candidates
        .select(pl.col('diseaseIds').alias('diseaseId'), pl.exclude('diseaseIds'))
        .explode('diseaseId')
        .filter(pl.col('diseaseId').is_not_null(), pl.col('diseaseId') != '')
        .sort(
            'diseaseId',
            'ldAncestry',
            'excludedFromRelease',
            'nSignificantCredibleSets',
            'h2ZScore',
            'studyId',
            descending=[False, False, False, True, True, False],
            nulls_last=True,
        )
        .unique(['diseaseId', 'ldAncestry'], keep='first', maintain_order=True)
        .select(list(OUTPUT_SCHEMA))
    )


def representative_changes(current: pl.DataFrame, previous: pl.DataFrame | None) -> pl.DataFrame:
    """Compare each disease and ancestry's representative with the previous run's, in `CHANGES_SCHEMA`."""
    cells = ['diseaseId', 'ldAncestry']
    if previous is None:
        previous = pl.DataFrame(schema={k: OUTPUT_SCHEMA[k] for k in [*cells, 'studyId']})
    return (
        current
        .select(*cells, 'studyId')
        .join(
            previous.select(*cells, pl.col('studyId').alias('previousStudyId')),
            on=cells,
            how='full',
            coalesce=True,
        )
        .with_columns(
            pl
            .when(pl.col('previousStudyId').is_null())
            .then(pl.lit('added'))
            .when(pl.col('studyId').is_null())
            .then(pl.lit('dropped'))
            .when(pl.col('studyId') == pl.col('previousStudyId'))
            .then(pl.lit('unchanged'))
            .otherwise(pl.lit('replaced'))
            .alias('change')
        )
        .select(list(CHANGES_SCHEMA))
        .cast(CHANGES_SCHEMA)
        .sort(cells)
    )


def representatives(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Choose the representative studies, and compare them with the previous run's.

    Args:
        source: the `RepresentativesSource` paths.
        destination: the `RepresentativesDestination` paths.
        settings: the `RepresentativesSettings`.
        config: unused.

    Raises:
        ValueError: if `include_excluded_studies` is set without an `excluded_study` source.
    """
    sources = RepresentativesSource.model_validate(source)
    destinations = RepresentativesDestination.model_validate(destination)
    options = RepresentativesSettings.model_validate(settings)
    check_destinations_absent([destinations.representatives, destinations.changes])
    studies = eligible_studies(scan_dataset(sources.study), options)
    if options.include_excluded_studies:
        if sources.excluded_study is None:
            msg = 'include_excluded_studies needs the excluded_study source'
            raise ValueError(msg)
        excluded = eligible_studies(scan_dataset(sources.excluded_study), options, excluded=True)
        studies = pl.concat([studies, excluded.join(studies.select('studyId'), on='studyId', how='anti')])
    heritability = successful_heritability(scan_dataset(sources.heritability))
    loci = significant_loci(scan_dataset(sources.credible_set), options.max_lead_p_value)
    candidates = (
        heritable_studies(studies, heritability, options)
        .join(loci, on='studyId', how='left')
        .with_columns(pl.col('nSignificantCredibleSets').fill_null(pl.when(~pl.col('excludedFromRelease')).then(0)))
    )
    chosen = choose_representatives(candidates).collect().cast(OUTPUT_SCHEMA)
    logger.info(
        '{} representative studies over {} disease and ancestry cells, {} of them excluded from the release',
        chosen['studyId'].n_unique(),
        chosen.height,
        chosen['excludedFromRelease'].sum(),
    )
    previous = previous_path(sources.previous_output, destinations.representatives)
    changes = representative_changes(chosen, None if previous is None else scan_dataset(previous).collect())
    logger.info('changes since the previous run: {}', dict(changes.group_by('change').len().iter_rows()))
    write_dataset(chosen, destinations.representatives)
    write_dataset(changes, destinations.changes)
