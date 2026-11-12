"""VIDRA step 2: the Bayesian dose-response fit of every gene-disease pair.

Port of VIDRA's `run_bayesian_analysis.py`. The analysis-ready table of `vidra_input` is
preprocessed per gene (annotation imputation by consequence class, deduplication, the lone
coding-GWAS filter) and then every gene-disease pair is fitted with one of two Stan models,
`VIDRA` (several variants) or `VIDRA_single_variant`, by ADVI. Both stages are grouped Arrow UDFs
(`applyInArrow`) whose groups are handled as polars frames. The output has one row per pair
and parameter with the posterior summary of that parameter.

Pairs that cannot identify a slope (a single ClinVar variant, a single rare variant without a
burden test) get a `single_variant_unidentified` row with no posterior; a pair whose fit fails
gets an `error_report` row with the error.
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from collections.abc import Callable
from typing import Any

import numpy as np
import polars as pl
import pyarrow as pa
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.pandas.types import to_arrow_schema

from pts.pyspark.common.session import Session
from pts.pyspark.vidra_utils.annotation import CLASS_LOF, CLASS_MISSENSE, CLASS_OTHER, CLASS_UNKNOWN
from pts.pyspark.vidra_utils.constants import (
    DEFAULT_BOSE,
    DEFAULT_XCSE,
    DEFAULT_YCSE,
    QTL_EQTL,
    QTL_NONE,
    QTL_PQTL,
    SOURCE_AZ_RARE,
    SOURCE_CLINVAR,
    SOURCE_CODING_GWAS,
    SOURCE_COMMON_QTL,
)
from pts.pyspark.vidra_utils.stan import (
    ADVI_DRAWS,
    ADVI_GRAD_SAMPLES,
    ADVI_MAX_ITER,
    ADVI_SEED,
    FULLRANK_MAX_N,
    MULTI_VARIANT,
    POSTERIOR_COLUMNS,
    SINGLE_VARIANT,
    load_models,
    scratch_dir,
    summarise,
)

SCORE_COLUMNS = ('as_gpnstar', 'as_alphamissense')
"""Protein-function scores imputed per gene; 1 is benign, 0 damaging."""
BENIGN = 1.0
DAMAGING = 0.0

SLOPE_BY_SOURCE = {
    'slope_eqtl': (SOURCE_COMMON_QTL, QTL_EQTL),
    'slope_pqtl': (SOURCE_COMMON_QTL, QTL_PQTL),
    'slope_coding_gwas': (SOURCE_CODING_GWAS, QTL_NONE),
    'slope_az_rare': (SOURCE_AZ_RARE, QTL_NONE),
    'slope_clinvar': (SOURCE_CLINVAR, QTL_NONE),
}
"""The per-source slopes of `VIDRA.stan`, `slope_random[1..5]` in this order, by their (GsourceLab, GqtlLab)."""
SLOPE_RANDOM_NAMES = tuple(SLOPE_BY_SOURCE)

VARIANT_PARAMETERS = ('xcest', 'yORest', 'protein_prior', 'disease_prior')
"""Per-variant latent parameters of both models, extracted with `save_variant_outputs`."""

META_COLUMNS = (
    'targetId',
    'diseaseId',
    'parameter',
    'model',
    'variantCount',
    'sources',
    'qtlTypes',
    'hasBurden',
    'error',
)
RESULT_COLUMNS = (*META_COLUMNS, *POSTERIOR_COLUMNS)
VARIANT_RESULT_COLUMNS = (*META_COLUMNS, 'variantId', *POSTERIOR_COLUMNS)

MODEL_SINGLE = 'single_variant'
MODEL_UNIDENTIFIED = 'single_variant_unidentified'
MODEL_ERROR = 'error_report'


class AdviSettings(BaseModel):
    """CmdStan ADVI options, the legacy values by default."""

    model_config = ConfigDict(extra='forbid')

    seed: int = ADVI_SEED
    max_iter: int = ADVI_MAX_ITER
    grad_samples: int = ADVI_GRAD_SAMPLES
    draws: int = ADVI_DRAWS
    fullrank_max_n: int = FULLRANK_MAX_N


class VidraBayesSettings(BaseModel):
    """What to fit, with what, and what to write."""

    model_config = ConfigDict(extra='forbid')

    h1: float = 0.1
    """Hyperparameter of the single-variant model."""
    genes: str | None = None
    """A text file of gene ids (one per line) restricting the run; every gene otherwise."""
    stan_exe_dir: str | None = None
    """Directory of the prebuilt model executables on the workers; compiled from source without it."""
    cmdstan_dir: str | None = None
    """The CmdStan installation on the workers; cmdstanpy's default without it."""
    save_stan_inputs: bool = False
    """Also write the preprocessed rows, exactly what the models see, to `destination['stan_inputs']`."""
    save_variant_outputs: bool = False
    """Also fit every pair a second time to write per-variant posteriors to `destination['variant_results']`."""
    output_partitions: int = Field(default=200, gt=0)
    advi: AdviSettings = Field(default_factory=AdviSettings)


# ----------------------------------------------------------------------------------------------
# Per-gene preprocessing
# ----------------------------------------------------------------------------------------------

DEDUP_KEY = ('variant', 'as_disease', 'GsourceLab', 'GqtlLab')
EFFECT_DEFAULTS = (
    ('xc', 0.0),
    ('xcse', DEFAULT_XCSE),
    ('yc', 0.0),
    ('ycse', DEFAULT_YCSE),
    ('bO', 0.0),
    ('bOse', DEFAULT_BOSE),
    ('as_clinicalSignificance', 0.0),
)
"""Model inputs and the value a missing (null, NaN) one takes."""


def _number(column: str) -> pl.Expr:
    """A column as floats, with NaN as null."""
    return pl.col(column).cast(pl.Float64, strict=False).fill_nan(None)


def impute_scores(gene: pl.DataFrame) -> pl.DataFrame:
    """Fill the protein-function scores of a gene's rows by their consequence class.

    - `lof`: both scores are set damaging (0), whatever was observed;
    - `missense`, `inframe_or_splice`, `unknown`: a missing score takes the gene's mean over its
      missense rows with a score, else over all its rows with a score, else benign;
    - `other` (synonymous, UTR, intronic, non-coding): a missing score is benign.

    Examples:
        >>> gene = pl.DataFrame({
        ...     'consequence_class': ['missense', 'missense', 'lof', 'other', 'unknown'],
        ...     'as_gpnstar': [0.2, None, 0.9, None, None],
        ...     'as_alphamissense': [0.4, 0.6, None, None, None],
        ... })
        >>> impute_scores(gene).select('as_gpnstar', 'as_alphamissense').rows()
        [(0.2, 0.4), (0.2, 0.6), (0.0, 0.0), (1.0, 1.0), (0.2, 0.5)]
    """
    classes = pl.col('consequence_class').cast(pl.String).fill_null(CLASS_UNKNOWN)
    imputed = []
    for column in SCORE_COLUMNS:
        score = _number(column)
        means = gene.select(score.filter(classes == CLASS_MISSENSE).mean(), score.mean().alias('all')).row(0)
        fill = next((mean for mean in means if mean is not None), BENIGN)
        imputed.append(
            pl
            .when(classes == CLASS_LOF)
            .then(DAMAGING)
            .when(score.is_null() & (classes == CLASS_OTHER))
            .then(BENIGN)
            .otherwise(score.fill_null(fill))
            .alias(column)
        )
    return gene.with_columns(imputed)


def deduplicate(gene: pl.DataFrame) -> pl.DataFrame:
    """One row per variant, disease and source: the largest `|yc|`, then the highest clinical significance.

    Rows keep their order.

    Examples:
        >>> gene = pl.DataFrame({
        ...     'variant': ['v', 'v', 'v'], 'as_disease': ['d'] * 3, 'GsourceLab': [2, 2, 0], 'GqtlLab': [2, 2, 0],
        ...     'yc': [0.0, 0.0, 0.5], 'as_clinicalSignificance': [0.5, 1.0, 0.0],
        ... })
        >>> deduplicate(gene).select('GsourceLab', 'as_clinicalSignificance').rows()
        [(2, 1.0), (0, 0.0)]
    """
    return (
        gene
        .with_row_index('_row')
        .sort(pl.col('yc').abs(), 'as_clinicalSignificance', nulls_last=True, maintain_order=True)
        .unique(subset=list(DEDUP_KEY), keep='last', maintain_order=True)
        .sort('_row')
        .drop('_row')
    )


def drop_lone_coding_variants(gene: pl.DataFrame) -> pl.DataFrame:
    """Drop a coding-GWAS variant that is alone in its disease and source group.

    Without an identified intercept a lone coding variant cannot identify its slope, yet the
    multi-variant model would still pool it. Kept from the original pipeline.
    """
    alone = pl.len().over('as_disease', 'GsourceLab', 'GqtlLab') == 1
    return gene.filter(~(alone & (pl.col('GsourceLab') == SOURCE_CODING_GWAS)))


def preprocess_gene(gene: pl.DataFrame) -> pl.DataFrame:
    """Impute, deduplicate and filter one gene's rows; the columns are unchanged."""
    if gene.is_empty():
        return gene
    out = impute_scores(gene).with_columns(
        *[_number(column).fill_null(default).alias(column) for column, default in EFFECT_DEFAULTS],
        pl.col('GsourceLab', 'GqtlLab').cast(pl.Int64),
    )
    return drop_lone_coding_variants(deduplicate(out))


# ----------------------------------------------------------------------------------------------
# Fitting one gene-disease pair
# ----------------------------------------------------------------------------------------------


def _finite(values: pl.Series, default: float, floor: float | None = None) -> list[float]:
    """Floats with every missing or infinite value replaced by `default`, optionally clipped below."""
    numbers = values.cast(pl.Float64, strict=False).to_numpy().astype(float)
    numbers = np.where(np.isfinite(numbers), numbers, default)
    if floor is not None:
        numbers = np.maximum(numbers, floor)
    return numbers.tolist()


def _first(pair: pl.DataFrame, column: str, default: float, floor: float | None = None) -> float:
    return _finite(pair[column].head(1), default, floor)[0]


def _meta(pair: pl.DataFrame, model: str, **extra: Any) -> dict[str, Any]:
    return {
        'targetId': str(pair['as_gene'][0]),
        'diseaseId': str(pair['as_disease'][0]),
        'model': model,
        'variantCount': pair['variant'].n_unique(),
        'sources': sorted({int(x) for x in pair['GsourceLab']}),
        'qtlTypes': sorted({int(x) for x in pair['GqtlLab']}),
        'hasBurden': _first(pair, 'bO', 0.0) != 0.0,
        'error': None,
        **extra,
    }


def _draws(fit: Any) -> dict[str, np.ndarray]:
    """The draws of a cmdstanpy variational fit, by Stan column name (`slope`, `slope_random[1]`, ...)."""
    return dict(zip(fit.column_names, np.asarray(fit.variational_sample).T, strict=True))


def _variational(
    model: Any, data: dict[str, Any], settings: VidraBayesSettings, algorithm: str
) -> dict[str, np.ndarray]:
    """Run ADVI in a scratch directory that is removed afterwards, and return the draws."""
    output_dir = tempfile.mkdtemp(prefix='vidra_advi_', dir=scratch_dir())
    try:
        fit = model.variational(
            data=data,
            seed=settings.advi.seed,
            algorithm=algorithm,
            iter=settings.advi.max_iter,
            grad_samples=settings.advi.grad_samples,
            draws=settings.advi.draws,
            require_converged=False,
            show_console=False,
            output_dir=output_dir,
        )
        return _draws(fit)
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)


def _variant_rows(draws: dict[str, np.ndarray], pair: pl.DataFrame, vector: bool) -> list[dict[str, Any]]:
    """Posterior summaries of the per-variant latent parameters, one row per variant and parameter."""
    rows = []
    for position, variant in enumerate(pair.iter_rows(named=True), start=1):
        for parameter in VARIANT_PARAMETERS:
            column = f'{parameter}[{position}]' if vector else parameter
            if column not in draws:
                continue
            rows.append({
                **summarise(draws[column]),
                'parameter': parameter,
                'variantId': str(variant['variant']),
                'sources': [int(variant['GsourceLab'])],
                'qtlTypes': [int(variant['GqtlLab'])],
            })
    return rows


def fit_single_variant(pair: pl.DataFrame, settings: VidraBayesSettings, variant_outputs: bool) -> list[dict[str, Any]]:
    """Fit `VIDRA_single_variant` to a one-variant pair, or report it unidentified.

    A single ClinVar variant has no measured effect, and a single rare variant without a burden
    test cannot separate its slope from the intercept: both get `single_variant_unidentified`.
    """
    source, qtl = int(pair['GsourceLab'][0]), int(pair['GqtlLab'][0])
    burden = _first(pair, 'bO', 0.0)
    has_burden = burden != 0.0
    if source == SOURCE_CLINVAR or (source == SOURCE_AZ_RARE and not has_burden):
        return [] if variant_outputs else [_meta(pair, MODEL_UNIDENTIFIED, parameter='slope')]
    data = {
        'h1': settings.h1,
        'N': 1,
        'numG1': float(source),
        'numG2': float(qtl),
        'bO': burden,
        'bOse': _first(pair, 'bOse', DEFAULT_BOSE, floor=1e-6),
        'has_burden': int(has_burden),
        'xc': _first(pair, 'xc', 0.0),
        'xcse': _first(pair, 'xcse', DEFAULT_XCSE, floor=1e-6),
        'yOR': _first(pair, 'yc', 0.0),
        'yORse': _first(pair, 'ycse', DEFAULT_YCSE, floor=1e-6),
        'as_gpnstar': _first(pair, 'as_gpnstar', BENIGN),
        'as_alphamissense': _first(pair, 'as_alphamissense', BENIGN),
        'as_clinicalSignificance': _first(pair, 'as_clinicalSignificance', 0.0),
    }
    model = load_models(settings.stan_exe_dir, settings.cmdstan_dir)[SINGLE_VARIANT]
    draws = _variational(model, data, settings, 'fullrank')
    meta = _meta(pair, MODEL_SINGLE)
    if variant_outputs:
        return [{**meta, **row} for row in _variant_rows(draws, pair, vector=False)]
    return [{**meta, **summarise(draws['slope']), 'parameter': 'slope'}]


def fit_multi_variant(pair: pl.DataFrame, settings: VidraBayesSettings, variant_outputs: bool) -> list[dict[str, Any]]:
    """Fit `VIDRA` to a pair with several variants.

    Returns the shared slope, the intercept, the per-source slopes and their pool.

    `meta_slope` pools the draws of the per-source slopes of the sources present in the pair.
    Fullrank ADVI up to `fullrank_max_n` variants, meanfield above.
    """
    data = {
        'nu': max(pair['variant'].n_unique() - 1, 1),
        'N': pair.height,
        'numG1': pair['GsourceLab'].cast(pl.Float64).to_list(),
        'numG2': pair['GqtlLab'].cast(pl.Float64).to_list(),
        'xc': _finite(pair['xc'], 1.0),
        'xcse': _finite(pair['xcse'], 1.0, floor=1e-6),
        'yOR': _finite(pair['yc'], 0.0),
        'yORse': _finite(pair['ycse'], DEFAULT_YCSE, floor=1e-6),
        'bO': _finite(pair['bO'], 0.0),
        'bOse': _finite(pair['bOse'], DEFAULT_BOSE),
        'as_gpnstar': _finite(pair['as_gpnstar'], BENIGN),
        'as_alphamissense': _finite(pair['as_alphamissense'], BENIGN),
        'as_clinicalSignificance': _finite(pair['as_clinicalSignificance'], 0.0),
    }
    algorithm = 'fullrank' if pair.height <= settings.advi.fullrank_max_n else 'meanfield'
    model = load_models(settings.stan_exe_dir, settings.cmdstan_dir)[MULTI_VARIANT]
    draws = _variational(model, data, settings, algorithm)
    meta = _meta(pair, f'multiple_variant_{algorithm}')
    if variant_outputs:
        return [{**meta, **row} for row in _variant_rows(draws, pair, vector=True)]
    parameters = {'slope': draws['slope'], 'intercept': draws['intercept_random']}
    for index, name in enumerate(SLOPE_RANDOM_NAMES, start=1):
        parameters[name] = draws[f'slope_random[{index}]']
    present = {(int(source), int(qtl)) for source, qtl in pair.select('GsourceLab', 'GqtlLab').unique().rows()}
    pooled = [parameters[name] for name, key in SLOPE_BY_SOURCE.items() if key in present]
    parameters['meta_slope'] = np.concatenate(pooled) if pooled else parameters['slope']
    return [{**meta, **summarise(values), 'parameter': name} for name, values in parameters.items()]


def fit_pair(pair: pl.DataFrame, settings: VidraBayesSettings, variant_outputs: bool = False) -> list[dict[str, Any]]:
    """The result rows of one gene-disease pair; a failed fit is one `error_report` row (none for variant outputs)."""
    try:
        if pair['variant'].n_unique() > 1:
            return fit_multi_variant(pair, settings, variant_outputs)
        return fit_single_variant(pair, settings, variant_outputs)
    except Exception as error:  # every failure of a pair is reported as a row, never raised
        message = f'{type(error).__name__}: {error} || {traceback.format_exc()[-400:]}'
        logger.warning('VIDRA fit failed for {} / {}: {}', pair['as_gene'][0], pair['as_disease'][0], message)
        return [] if variant_outputs else [_meta(pair, MODEL_ERROR, parameter='slope', error=message[:2000])]


def result_schema(variant_outputs: bool = False) -> t.StructType:
    """The Spark schema of the result rows."""
    fields = [
        t.StructField('targetId', t.StringType()),
        t.StructField('diseaseId', t.StringType()),
        t.StructField('parameter', t.StringType()),
        t.StructField('model', t.StringType()),
        t.StructField('variantCount', t.IntegerType()),
        t.StructField('sources', t.ArrayType(t.IntegerType())),
        t.StructField('qtlTypes', t.ArrayType(t.IntegerType())),
        t.StructField('hasBurden', t.BooleanType()),
        t.StructField('error', t.StringType()),
    ]
    if variant_outputs:
        fields.append(t.StructField('variantId', t.StringType()))
    fields += [t.StructField(column, t.DoubleType()) for column in POSTERIOR_COLUMNS]
    return t.StructType(fields)


def _polars_schema(variant_outputs: bool) -> dict[str, pl.DataType]:
    integers = pl.List(pl.Int32)
    types: dict[str, pl.DataType] = {
        'variantCount': pl.Int32(),
        'sources': integers,
        'qtlTypes': integers,
        'hasBurden': pl.Boolean(),
        **{column: pl.Float64() for column in POSTERIOR_COLUMNS},
    }
    columns = VARIANT_RESULT_COLUMNS if variant_outputs else RESULT_COLUMNS
    return {column: types.get(column, pl.String()) for column in columns}


def process_pair(pair: pl.DataFrame, settings: VidraBayesSettings, variant_outputs: bool = False) -> pl.DataFrame:
    """The result rows of one pair as a frame with every output column, in `result_schema` order."""
    schema = _polars_schema(variant_outputs)
    if pair.is_empty():
        return pl.DataFrame(schema=schema)
    return pl.from_dicts(fit_pair(pair, settings, variant_outputs), schema=schema)


def arrow_udf(
    function: Callable[[pl.DataFrame], pl.DataFrame], schema: pa.Schema | None = None
) -> Callable[[pa.Table], pa.Table]:
    """A polars frame function as an `applyInArrow` function.

    The result is cast to `schema`, or to the input's schema without one, so that polars' large
    string and list types become the types Spark expects.
    """

    def run(table: pa.Table) -> pa.Table:
        result = function(pl.DataFrame(table)).to_arrow()
        return result.cast(schema or table.schema)

    return run


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def restrict_genes(rows: DataFrame, genes: list[str]) -> DataFrame:
    """The rows of `genes` only, as a literal filter so that Spark prunes the gene partitions."""
    return rows.filter(f.col('as_gene').isin(genes))


def fit_pairs(preprocessed: DataFrame, pairs: int, settings: VidraBayesSettings, variant_outputs: bool) -> DataFrame:
    """Every gene-disease pair fitted on the workers, one pair per Spark partition."""
    schema = result_schema(variant_outputs)
    return (
        preprocessed
        .repartition(max(pairs, 1), 'as_gene', 'as_disease')
        .groupby('as_gene', 'as_disease')
        .applyInArrow(
            arrow_udf(lambda pair: process_pair(pair, settings, variant_outputs), to_arrow_schema(schema)),
            schema=schema,
        )
    )


def vidra_bayes(source: str, destination: dict[str, str], settings: dict[str, Any], properties: dict[str, str]) -> None:
    """Fit every gene-disease pair of the analysis-ready table and write the posterior summaries.

    Args:
        source: the `vidra_input` output, partitioned by `as_gene`.
        destination: `results`, and with the matching settings `stan_inputs` and `variant_results`.
        settings: `VidraBayesSettings`.
        properties: Spark properties.
    """
    options = VidraBayesSettings.model_validate(settings)
    session = Session(app_name='vidra_bayes', properties=properties)
    spark = session.spark

    rows = spark.read.parquet(source)
    if options.genes:
        genes = [line.value.strip() for line in spark.read.text(options.genes).collect() if line.value.strip()]
        logger.info('restricting the run to {} genes from {}', len(genes), options.genes)
        rows = restrict_genes(rows, genes)

    preprocessed = rows.groupby('as_gene').applyInArrow(arrow_udf(preprocess_gene), schema=rows.schema).cache()
    pairs = preprocessed.select('as_gene', 'as_disease').distinct().count()
    logger.info('{} rows in {} gene-disease pairs after preprocessing', preprocessed.count(), pairs)
    if options.save_stan_inputs:
        preprocessed.write.mode('overwrite').partitionBy('as_gene').parquet(destination['stan_inputs'])
        logger.info('Stan inputs written to {}', destination['stan_inputs'])

    fitted = fit_pairs(preprocessed, pairs, options, variant_outputs=False)
    fitted.coalesce(options.output_partitions).write.mode('overwrite').parquet(destination['results'])
    logger.info(
        '{} result rows written to {}', spark.read.parquet(destination['results']).count(), destination['results']
    )

    if options.save_variant_outputs:
        variant_rows = fit_pairs(preprocessed, pairs, options, variant_outputs=True)
        variant_rows.coalesce(options.output_partitions).write.mode('overwrite').parquet(destination['variant_results'])
        logger.info('per-variant posteriors written to {}', destination['variant_results'])
