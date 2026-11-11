"""VIDRA step 2: the Bayesian dose-response fit of every gene-disease pair.

Port of VIDRA's `run_bayesian_analysis.py`. The analysis-ready table of `vidra_input` is
preprocessed per gene (annotation imputation by consequence class, deduplication, the lone
coding-GWAS filter) and then every gene-disease pair is fitted with one of two Stan models,
`VIDRA` (several variants) or `VIDRA_single_variant`, by ADVI. The output has one row per pair
and parameter with the posterior summary of that parameter.

Pairs that cannot identify a slope (a single ClinVar variant, a single rare variant without a
burden test) get a `single_variant_unidentified` row with no posterior; a pair whose fit fails
gets an `error_report` row with the error.
"""

from __future__ import annotations

import shutil
import tempfile
import traceback
from typing import Any

import numpy as np
import pandas as pd
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.common.session import Session
from pts.pyspark.vidra_utils.annotation import CLASS_LOF, CLASS_MISSENSE, CLASS_OTHER
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


def impute_scores(gene: pd.DataFrame) -> pd.DataFrame:
    """Fill the protein-function scores of a gene's rows by their consequence class.

    - `lof`: both scores are set damaging (0), whatever was observed;
    - `missense`, `inframe_or_splice`, `unknown`: a missing score takes the gene's mean over its
      missense rows with a score, else over all its rows with a score, else benign;
    - `other` (synonymous, UTR, intronic, non-coding): a missing score is benign.

    Examples:
        >>> gene = pd.DataFrame({
        ...     'consequence_class': ['missense', 'missense', 'lof', 'other', 'unknown'],
        ...     'as_gpnstar': [0.2, None, 0.9, None, None],
        ...     'as_alphamissense': [0.4, 0.6, None, None, None],
        ... })
        >>> impute_scores(gene)[['as_gpnstar', 'as_alphamissense']].to_numpy().tolist()
        [[0.2, 0.4], [0.2, 0.6], [0.0, 0.0], [1.0, 1.0], [0.2, 0.5]]
    """
    out = gene.copy()
    classes = out['consequence_class'].fillna('unknown')
    for column in SCORE_COLUMNS:
        values = pd.to_numeric(out[column], errors='coerce')
        missense_mean = values[classes == CLASS_MISSENSE].mean()
        gene_mean = missense_mean if pd.notna(missense_mean) else values.mean()
        fill = gene_mean if pd.notna(gene_mean) else BENIGN
        values = values.where(~((classes == CLASS_OTHER) & values.isna()), BENIGN)
        values = values.fillna(fill)
        values[classes == CLASS_LOF] = DAMAGING
        out[column] = values.astype(float)
    return out


def deduplicate(gene: pd.DataFrame) -> pd.DataFrame:
    """One row per variant, disease and source: the largest `|yc|`, then the highest clinical significance.

    Examples:
        >>> gene = pd.DataFrame({
        ...     'variant': ['v', 'v', 'v'], 'as_disease': ['d'] * 3, 'GsourceLab': [2, 2, 0], 'GqtlLab': [2, 2, 0],
        ...     'yc': [0.0, 0.0, 0.5], 'as_clinicalSignificance': [0.5, 1.0, 0.0],
        ... })
        >>> deduplicate(gene)[['GsourceLab', 'as_clinicalSignificance']].to_numpy().tolist()
        [[2, 1.0], [0, 0.0]]
    """
    ordered = gene.assign(_abs_yc=gene['yc'].abs()).sort_values(['_abs_yc', 'as_clinicalSignificance'], kind='stable')
    kept = ordered.drop_duplicates(subset=['variant', 'as_disease', 'GsourceLab', 'GqtlLab'], keep='last')
    return kept.drop(columns='_abs_yc').sort_index()


def drop_lone_coding_variants(gene: pd.DataFrame) -> pd.DataFrame:
    """Drop a coding-GWAS variant that is alone in its disease and source group.

    Without an identified intercept a lone coding variant cannot identify its slope, yet the
    multi-variant model would still pool it. Kept from the original pipeline.
    """
    group = gene.groupby(['as_disease', 'GsourceLab', 'GqtlLab'])['variant'].transform('size')
    lone_coding = (group == 1) & (gene['GsourceLab'] == SOURCE_CODING_GWAS)
    return gene[~lone_coding]


def preprocess_gene(gene: pd.DataFrame) -> pd.DataFrame:
    """Impute, deduplicate and filter one gene's rows; the columns are unchanged."""
    if gene.empty:
        return gene
    out = impute_scores(gene)
    for column, default in (
        ('xc', 0.0),
        ('xcse', DEFAULT_XCSE),
        ('yc', 0.0),
        ('ycse', DEFAULT_YCSE),
        ('bO', 0.0),
        ('bOse', DEFAULT_BOSE),
    ):
        out[column] = pd.to_numeric(out[column], errors='coerce').fillna(default)
    out['as_clinicalSignificance'] = pd.to_numeric(out['as_clinicalSignificance'], errors='coerce').fillna(0.0)
    out['GsourceLab'] = out['GsourceLab'].astype(int)
    out['GqtlLab'] = out['GqtlLab'].astype(int)
    return drop_lone_coding_variants(deduplicate(out))


# ----------------------------------------------------------------------------------------------
# Fitting one gene-disease pair
# ----------------------------------------------------------------------------------------------


def _finite(values: pd.Series, default: float, floor: float | None = None) -> list[float]:
    numbers = pd.to_numeric(values, errors='coerce').replace([np.inf, -np.inf], np.nan).fillna(default)
    if floor is not None:
        numbers = numbers.clip(lower=floor)
    return numbers.astype(float).tolist()


def _meta(pair: pd.DataFrame, model: str, **extra: Any) -> dict[str, Any]:
    return {
        'targetId': str(pair['as_gene'].iloc[0]),
        'diseaseId': str(pair['as_disease'].iloc[0]),
        'model': model,
        'variantCount': int(pair['variant'].nunique()),
        'sources': sorted({int(x) for x in pair['GsourceLab']}),
        'qtlTypes': sorted({int(x) for x in pair['GqtlLab']}),
        'hasBurden': bool(float(pair['bO'].iloc[0]) != 0.0),
        'error': None,
        **extra,
    }


def _variant_rows(draws: pd.DataFrame, pair: pd.DataFrame, vector: bool) -> list[dict[str, Any]]:
    """Posterior summaries of the per-variant latent parameters, one row per variant and parameter."""
    rows = []
    for position, (_, variant) in enumerate(pair.iterrows(), start=1):
        for parameter in VARIANT_PARAMETERS:
            column = f'{parameter}[{position}]' if vector else parameter
            if column not in draws.columns:
                continue
            rows.append({
                **summarise(draws[column].to_numpy()),
                'parameter': parameter,
                'variantId': str(variant['variant']),
                'sources': [int(variant['GsourceLab'])],
                'qtlTypes': [int(variant['GqtlLab'])],
            })
    return rows


def fit_single_variant(pair: pd.DataFrame, settings: VidraBayesSettings, variant_outputs: bool) -> list[dict[str, Any]]:
    """Fit `VIDRA_single_variant` to a one-variant pair, or report it unidentified.

    A single ClinVar variant has no measured effect, and a single rare variant without a burden
    test cannot separate its slope from the intercept: both get `single_variant_unidentified`.
    """
    row = pair.iloc[0]
    source, qtl = int(row['GsourceLab']), int(row['GqtlLab'])
    burden = float(_finite(pd.Series([row['bO']]), 0.0)[0])
    has_burden = burden != 0.0
    if source == SOURCE_CLINVAR or (source == SOURCE_AZ_RARE and not has_burden):
        return [] if variant_outputs else [_meta(pair, MODEL_UNIDENTIFIED, parameter='slope')]
    data = {
        'h1': settings.h1,
        'N': 1,
        'numG1': float(source),
        'numG2': float(qtl),
        'bO': burden,
        'bOse': max(_finite(pd.Series([row['bOse']]), DEFAULT_BOSE)[0], 1e-6),
        'has_burden': int(has_burden),
        'xc': _finite(pd.Series([row['xc']]), 0.0)[0],
        'xcse': max(_finite(pd.Series([row['xcse']]), DEFAULT_XCSE)[0], 1e-6),
        'yOR': _finite(pd.Series([row['yc']]), 0.0)[0],
        'yORse': max(_finite(pd.Series([row['ycse']]), DEFAULT_YCSE)[0], 1e-6),
        'as_gpnstar': _finite(pd.Series([row['as_gpnstar']]), BENIGN)[0],
        'as_alphamissense': _finite(pd.Series([row['as_alphamissense']]), BENIGN)[0],
        'as_clinicalSignificance': _finite(pd.Series([row['as_clinicalSignificance']]), 0.0)[0],
    }
    model = load_models(settings.stan_exe_dir, settings.cmdstan_dir)[SINGLE_VARIANT]
    with tempfile.TemporaryDirectory(prefix='vidra_sv_', dir=scratch_dir()) as output_dir:
        fit = model.variational(
            data=data,
            seed=settings.advi.seed,
            algorithm='fullrank',
            iter=settings.advi.max_iter,
            grad_samples=settings.advi.grad_samples,
            draws=settings.advi.draws,
            require_converged=False,
            show_console=False,
            output_dir=output_dir,
        )
        draws = fit.variational_sample_pd
    meta = _meta(pair, MODEL_SINGLE)
    if variant_outputs:
        return [{**meta, **row} for row in _variant_rows(draws, pair, vector=False)]
    return [{**meta, **summarise(draws['slope'].to_numpy()), 'parameter': 'slope'}]


def fit_multi_variant(pair: pd.DataFrame, settings: VidraBayesSettings, variant_outputs: bool) -> list[dict[str, Any]]:
    """Fit `VIDRA` to a pair with several variants.

    Returns the shared slope, the intercept, the per-source slopes and their pool.

    `meta_slope` pools the draws of the per-source slopes of the sources present in the pair.
    Fullrank ADVI up to `fullrank_max_n` variants, meanfield above.
    """
    data = {
        'nu': max(int(pair['variant'].nunique()) - 1, 1),
        'N': len(pair),
        'numG1': pair['GsourceLab'].astype(float).tolist(),
        'numG2': pair['GqtlLab'].astype(float).tolist(),
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
    algorithm = 'fullrank' if len(pair) <= settings.advi.fullrank_max_n else 'meanfield'
    model = load_models(settings.stan_exe_dir, settings.cmdstan_dir)[MULTI_VARIANT]
    output_dir = tempfile.mkdtemp(prefix='vidra_mv_', dir=scratch_dir())
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
        draws = fit.variational_sample_pd
    finally:
        shutil.rmtree(output_dir, ignore_errors=True)
    meta = _meta(pair, f'multiple_variant_{algorithm}')
    if variant_outputs:
        return [{**meta, **row} for row in _variant_rows(draws, pair, vector=True)]
    parameters = {'slope': draws['slope'].to_numpy(), 'intercept': draws['intercept_random'].to_numpy()}
    for index, name in enumerate(SLOPE_RANDOM_NAMES, start=1):
        parameters[name] = draws[f'slope_random[{index}]'].to_numpy()
    labels = pair[['GsourceLab', 'GqtlLab']].drop_duplicates().itertuples(index=False)
    present = {(int(source), int(qtl)) for source, qtl in labels}
    pooled = [parameters[name] for name, key in SLOPE_BY_SOURCE.items() if key in present]
    parameters['meta_slope'] = np.concatenate(pooled) if pooled else parameters['slope']
    return [{**meta, **summarise(values), 'parameter': name} for name, values in parameters.items()]


def fit_pair(pair: pd.DataFrame, settings: VidraBayesSettings, variant_outputs: bool = False) -> list[dict[str, Any]]:
    """The result rows of one gene-disease pair; a failed fit is one `error_report` row (none for variant outputs)."""
    try:
        if pair['variant'].nunique() > 1:
            return fit_multi_variant(pair, settings, variant_outputs)
        return fit_single_variant(pair, settings, variant_outputs)
    except Exception as error:  # every failure of a pair is reported as a row, never raised
        message = f'{type(error).__name__}: {error} || {traceback.format_exc()[-400:]}'
        logger.warning('VIDRA fit failed for {} / {}: {}', pair['as_gene'].iloc[0], pair['as_disease'].iloc[0], message)
        return [] if variant_outputs else [_meta(pair, MODEL_ERROR, parameter='slope', error=message[:2000])]


def _frame(rows: list[dict[str, Any]], columns: tuple[str, ...]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame[list(columns)]


def process_pair(pair: pd.DataFrame, settings: VidraBayesSettings, variant_outputs: bool = False) -> pd.DataFrame:
    """`fit_pair` as a pandas UDF: the rows of one pair with every output column."""
    columns = VARIANT_RESULT_COLUMNS if variant_outputs else RESULT_COLUMNS
    if pair.empty:
        return _frame([], columns)
    return _frame(fit_pair(pair, settings, variant_outputs), columns)


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


# ----------------------------------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------------------------------


def restrict_genes(rows: DataFrame, genes: list[str]) -> DataFrame:
    """The rows of `genes` only, as a literal filter so that Spark prunes the gene partitions."""
    return rows.filter(f.col('as_gene').isin(genes))


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

    preprocessed = rows.groupby('as_gene').applyInPandas(preprocess_gene, schema=rows.schema).cache()
    pairs = preprocessed.select('as_gene', 'as_disease').distinct().count()
    logger.info('{} rows in {} gene-disease pairs after preprocessing', preprocessed.count(), pairs)
    if options.save_stan_inputs:
        preprocessed.write.mode('overwrite').partitionBy('as_gene').parquet(destination['stan_inputs'])
        logger.info('Stan inputs written to {}', destination['stan_inputs'])

    fitted = (
        preprocessed
        .repartition(max(pairs, 1), 'as_gene', 'as_disease')
        .groupby('as_gene', 'as_disease')
        .applyInPandas(lambda pair: process_pair(pair, options), schema=result_schema())
    )
    fitted.coalesce(options.output_partitions).write.mode('overwrite').parquet(destination['results'])
    logger.info(
        '{} result rows written to {}', spark.read.parquet(destination['results']).count(), destination['results']
    )

    if options.save_variant_outputs:
        variant_rows = (
            preprocessed
            .repartition(max(pairs, 1), 'as_gene', 'as_disease')
            .groupby('as_gene', 'as_disease')
            .applyInPandas(lambda pair: process_pair(pair, options, variant_outputs=True), schema=result_schema(True))
        )
        variant_rows.coalesce(options.output_partitions).write.mode('overwrite').parquet(destination['variant_results'])
        logger.info('per-variant posteriors written to {}', destination['variant_results'])
