"""One representative study per disease and LD ancestry.

Most pairs of GWAS studies are two cohorts of the same disease, so genetic correlation is computed
between diseases, each stood for by its best powered study: the highest effective sample size,
among studies that are GWAS with summary statistics, pass QC, are of a kind LDSC applies to, have
one dominant LD ancestry with a score panel, and whose own heritability was estimated with
detectable signal.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.genetic_correlation.settings import LD_ANCESTRIES, RepresentativesSettings, RepresentativesSource
from pts.transformers.utils.dataset import scan_dataset, write_dataset

OUTPUT_SCHEMA = pl.Schema({
    'diseaseId': pl.String,
    'ldAncestry': pl.String,
    'studyId': pl.String,
    'effectiveSampleSize': pl.Float64,
    'nSamples': pl.Int32,
    'nCases': pl.Int32,
    'nControls': pl.Int32,
    'h2': pl.Float64,
    'h2StandardError': pl.Float64,
    'lambdaGc': pl.Float64,
})


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


def ld_ancestry(studies: pl.LazyFrame) -> pl.LazyFrame:
    """Each study's dominant LD population among those with a panel: `studyId` and `ldAncestry`.

    Populations are weighted by `relativeSampleSize`; a tie goes to `nfe`, then alphabetically.
    A study with no population among `LD_ANCESTRIES` has no row.
    """
    entry = pl.col('ldPopulationStructure')
    return (
        studies
        .select('studyId', 'ldPopulationStructure')
        .explode('ldPopulationStructure')
        .select(
            'studyId',
            entry.struct.field('ldPopulation').str.strip_chars().str.to_lowercase().alias('population'),
            entry.struct.field('relativeSampleSize').alias('weight'),
        )
        .filter(pl.col('population').is_in(LD_ANCESTRIES), pl.col('weight').is_not_null())
        .group_by('studyId', 'population')
        .agg(pl.col('weight').sum())
        .sort('studyId', 'weight', pl.col('population') == 'nfe', 'population', descending=[False, True, True, False])
        .unique('studyId', keep='first', maintain_order=True)
        .select('studyId', pl.col('population').alias('ldAncestry'))
    )


def eligible_studies(studies: pl.LazyFrame, options: RepresentativesSettings) -> pl.LazyFrame:
    """The studies a representative may be chosen from, with their ancestry and effective sample size."""
    flags = pl.col('analysisFlags').list.eval(pl.element().str.strip_chars().str.to_lowercase())
    compatible = pl.col('analysisFlags').is_null() | (
        flags.list.set_difference(options.allowed_analysis_flags).list.len() == 0
    )
    disqualified = (
        pl.col('qualityControls').fill_null([]).list.set_intersection(options.disqualifying_quality_controls).list.len()
        > 0
    )
    return (
        studies
        .filter(pl.col('studyType') == 'gwas', pl.col('hasSumstats'), ~disqualified, compatible)
        .with_columns(effective_sample_size())
        .filter(pl.col('effectiveSampleSize') >= options.min_effective_sample_size)
        .join(ld_ancestry(studies), on='studyId')
        .filter(pl.col('ldAncestry').is_in(options.ancestries))
        .select(
            'studyId',
            'ldAncestry',
            'effectiveSampleSize',
            pl.col('nSamples').cast(pl.Int32),
            pl.col('nCases').cast(pl.Int32),
            pl.col('nControls').cast(pl.Int32),
            pl.col(options.disease_column).alias('diseaseIds'),
        )
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
    """The eligible studies whose heritability estimate passes the thresholds, with `h2ZSquared`."""
    error = pl.when(pl.col('h2StandardError') > 0).then(pl.col('h2StandardError'))
    lambda_ok = pl.col('lambdaGc').is_null() | pl.col('lambdaGc').is_between(
        options.min_lambda_gc, options.max_lambda_gc
    )
    return (
        studies
        .join(heritability, on='studyId')
        .with_columns(((pl.col('h2') / error) ** 2).alias('h2ZSquared'))
        .filter(
            pl.col('h2') > options.min_h2,
            pl.col('h2') < options.max_h2,
            pl.col('h2ZSquared') >= options.min_h2_z_squared,
            lambda_ok,
        )
    )


def choose_representatives(candidates: pl.LazyFrame) -> pl.LazyFrame:
    """The best powered study per disease and ancestry: highest effective sample size, then `h2ZSquared`."""
    return (
        candidates
        .select(pl.col('diseaseIds').alias('diseaseId'), pl.exclude('diseaseIds'))
        .explode('diseaseId')
        .filter(pl.col('diseaseId').is_not_null(), pl.col('diseaseId') != '')
        .sort(
            'diseaseId',
            'ldAncestry',
            'effectiveSampleSize',
            'h2ZSquared',
            'studyId',
            descending=[False, False, True, True, False],
        )
        .unique(['diseaseId', 'ldAncestry'], keep='first', maintain_order=True)
        .select(list(OUTPUT_SCHEMA))
    )


def representatives(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Choose the representative studies.

    Args:
        source: the `RepresentativesSource` paths.
        destination: the representatives dataset, one row per disease and ancestry.
        settings: the `RepresentativesSettings`.
        config: unused.
    """
    sources = RepresentativesSource.model_validate(source)
    options = RepresentativesSettings.model_validate(settings)
    studies = eligible_studies(scan_dataset(sources.study), options)
    heritability = successful_heritability(scan_dataset(sources.heritability))
    chosen = choose_representatives(heritable_studies(studies, heritability, options)).collect()
    logger.info(
        '{} representative studies over {} disease and ancestry cells',
        chosen['studyId'].n_unique(),
        chosen.height,
    )
    write_dataset(chosen.cast(dict(OUTPUT_SCHEMA)), destination)
