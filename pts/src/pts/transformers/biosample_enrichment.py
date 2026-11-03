"""Biosample enrichment dataset generation.

For each GWAS study and each tissue or cell type in a specificity panel, tests whether the genes
L2G prioritises are more specific to that biosample than gene length and gene density explain.
The regression itself is in `pts.transformers.enrichment.regression`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.enrichment import regression
from pts.transformers.enrichment.regression import StandardErrors
from pts.transformers.utils.dataset import scan_dataset, write_dataset

BiosampleType = Literal['tissue', 'celltype']

_BIOSAMPLE_COLUMNS: dict[str, tuple[str, str]] = {
    # biosample type -> (column carrying the id, column that must be null)
    'tissue': ('tissueBiosampleId', 'celltypeBiosampleId'),
    'celltype': ('celltypeBiosampleId', 'tissueBiosampleId'),
}

OUTPUT_SCHEMA = pl.Schema({
    'studyId': pl.String,
    'datasourceId': pl.String,
    'biosampleType': pl.String,
    'biosampleId': pl.String,
    'nGenesScored': pl.Int32,
    'nStudiesPooled': pl.Int32,
    'beta': pl.Float64,
    'standardError': pl.Float64,
    'zScore': pl.Float64,
    'pValue': pl.Float64,
    'pValueBonferroni': pl.Float64,
    'qValue': pl.Float64,
})
"""Release schema. `nStudiesPooled` is 1 when studies are not pooled."""

_STATISTICS_SCHEMA = {
    name: OUTPUT_SCHEMA[name]
    for name in ('biosampleId', 'beta', 'standardError', 'zScore', 'pValue', 'pValueBonferroni', 'qValue')
}

_UNIT_BATCH = 512
"""Units fitted per pass, keeping the genes x units score matrix to about 80 MB."""


@dataclass(frozen=True)
class Panel:
    """One specificity panel: a datasource at one biosample resolution."""

    datasource_id: str
    biosample_type: BiosampleType

    @classmethod
    def from_settings(cls, entry: dict[str, str]) -> Panel:
        """Read a panel from its config entry.

        Args:
            entry: keys `datasourceId` and `biosampleType`.

        Returns:
            The panel.

        Raises:
            ValueError: if `biosampleType` is not `tissue` or `celltype`.
        """
        kind = entry['biosampleType']
        if kind not in _BIOSAMPLE_COLUMNS:
            raise ValueError(f'biosampleType must be one of {sorted(_BIOSAMPLE_COLUMNS)}, got {kind!r}')
        return cls(datasource_id=entry['datasourceId'], biosample_type=cast(BiosampleType, kind))


def gene_covariates(target: pl.LazyFrame, density_window: int) -> pl.DataFrame:
    """Gene length, TSS density and position per gene, for every gene with a usable location.

    Long genes in gene-poor regions are more often both GWAS hits and tissue specific, so these
    are adjusted for to stop that alone looking like enrichment.

    Args:
        target: the target index, with `id`, `biotype`, `tss` and `genomicLocation`.
        density_window: half-width, in base pairs, of the window around each TSS.

    Returns:
        `geneId`, `biotype`, `chromosome`, `tss`, `logLength` and `density`, one row per gene.
    """
    genes = (
        target
        .select(
            pl.col('id').alias('geneId'),
            'biotype',
            'tss',
            pl.col('genomicLocation').struct.field('chromosome'),
            (pl.col('genomicLocation').struct.field('end') - pl.col('genomicLocation').struct.field('start'))
            .abs()
            .alias('length'),
        )
        .filter(pl.col('length') > 0, pl.col('tss').is_not_null(), pl.col('chromosome').is_not_null())
        .unique('geneId', keep='first')
        .sort('geneId')
        .collect()
    )
    # TSSs within the window, the gene's own included, counting every biotype: any gene adds to
    # how crowded a region is
    density = np.empty(genes.height, dtype=np.int64)
    for _, rows in genes.with_row_index('row').group_by('chromosome'):
        tss = rows['tss'].to_numpy()
        ordered = np.sort(tss)
        upper = np.searchsorted(ordered, tss + density_window, side='right')
        lower = np.searchsorted(ordered, tss - density_window, side='left')
        density[rows['row'].to_numpy()] = upper - lower
    return genes.with_columns(
        pl.col('length').log10().alias('logLength'),
        pl.Series('density', density),
    ).drop('length')


def gwas_studies(study: pl.LazyFrame, pool_by_disease: bool) -> pl.LazyFrame:
    """Assign every GWAS study to the unit it is scored in.

    A unit is the study itself or, when pooling, every study with the same set of diseases.
    Studies with no diseases are not pooled.

    Args:
        study: the study index.
        pool_by_disease: pool studies with an identical `diseaseIds` set into one unit.

    Returns:
        `studyId` and `unitId`. Without pooling, `unitId` is the study id.
    """
    gwas = study.filter(pl.col('studyType') == 'gwas').select('studyId', 'diseaseIds')
    if not pool_by_disease:
        return gwas.select('studyId', pl.col('studyId').alias('unitId'))
    diseases = pl.col('diseaseIds').list.unique().list.sort().list.join(';')
    return gwas.select(
        'studyId',
        # prefixed so a disease set can never collide with a study id
        pl
        .when(pl.col('diseaseIds').list.len() > 0)
        .then(pl.lit('diseases:') + diseases)
        .otherwise(pl.lit('study:') + pl.col('studyId'))
        .alias('unitId'),
    )


def study_gene_scores(l2g: pl.LazyFrame, credible_set: pl.LazyFrame, units: pl.LazyFrame) -> pl.LazyFrame:
    """Highest L2G score per GWAS study and gene, over the study's credible sets.

    The release keeps only scores above its L2G threshold, so a missing gene is read as 0 later.

    Args:
        l2g: the L2G predictions.
        credible_set: the credible sets, for the study of each locus.
        units: `studyId` -> `unitId`, from `gwas_studies`; studies not in it are dropped.

    Returns:
        `studyId`, `unitId`, `geneId` and `score`.
    """
    loci = credible_set.select('studyLocusId', 'studyId').unique()
    return (
        l2g
        .select('studyLocusId', 'geneId', 'score')
        .join(loci, on='studyLocusId')
        .join(units, on='studyId')
        .group_by('studyId', 'unitId', 'geneId')
        .agg(pl.col('score').max())
    )


def specificity_matrix(baseline_expression: pl.LazyFrame, panel: Panel) -> pl.DataFrame:
    """A panel's specificity scores, one row per gene and one column per biosample.

    Args:
        baseline_expression: the baseline expression dataset.
        panel: the panel to read.

    Returns:
        `geneId` and then one Float64 column per biosample id, sorted. Missing values are null.

    Raises:
        ValueError: if the panel has no specificity scores.
    """
    id_column, null_column = _BIOSAMPLE_COLUMNS[panel.biosample_type]
    long = (
        baseline_expression
        .filter(
            pl.col('datasourceId') == panel.datasource_id,
            pl.col(id_column).is_not_null(),
            pl.col(null_column).is_null(),
            pl.col('specificity_score').is_not_null(),
        )
        # a gene and biosample repeat when several source ids map to one target, with equal scores
        .group_by(pl.col('targetId').alias('geneId'), pl.col(id_column).alias('biosampleId'))
        .agg(pl.col('specificity_score').max().cast(pl.Float64))
        .collect()
    )
    if long.is_empty():
        raise ValueError(f'no specificity scores for panel {panel}')
    wide = long.pivot(on='biosampleId', index='geneId', values='specificity_score')
    biosamples = sorted(c for c in wide.columns if c != 'geneId')
    return wide.select('geneId', *biosamples).sort('geneId')


def _standardise(values: np.ndarray) -> np.ndarray:
    spread = values.std(0)
    return (values - values.mean(0)) / np.where(spread > 0, spread, 1.0)


def score_panel(
    scores: pl.DataFrame,
    specificity: pl.DataFrame,
    covariates: pl.DataFrame,
    *,
    standard_errors: StandardErrors,
    region_window: int,
) -> pl.DataFrame:
    """Fit every unit against every biosample of one panel.

    Args:
        scores: `unitId`, `geneId` and `score`, the highest per unit and gene.
        specificity: the panel, from `specificity_matrix`.
        covariates: per-gene covariates, from `gene_covariates`.
        standard_errors: `region`, `gene` or `classical`; see `regression.Design`.
        region_window: window size, in base pairs, for `region` standard errors.

    Returns:
        `unitId`, `biosampleId`, `nGenesScored`, `beta`, `standardError`, `zScore`, `pValue`,
        `pValueBonferroni` and `qValue`. A unit with no scored gene in the universe has no rows.
    """
    # L2G only scores protein-coding genes; any other gene could never be prioritised
    universe = (
        covariates
        .filter(pl.col('biotype') == 'protein_coding')
        .join(specificity, on='geneId')
        .sort('geneId')
        .with_row_index('geneIndex')
    )
    biosamples = [c for c in specificity.columns if c != 'geneId']
    n_genes = universe.height
    logger.info(f'{n_genes} genes and {len(biosamples)} biosamples in the universe')

    x = universe.select(biosamples).fill_null(0.0).to_numpy().astype(np.float64)
    w = _standardise(universe.select('logLength', 'density').to_numpy().astype(np.float64))
    # genes in the same window share one association signal, so are counted as one for the error
    clusters = (
        regression.Clusters.from_labels(
            regression.region_labels(universe['chromosome'].to_numpy(), universe['tss'].to_numpy(), region_window)
        )
        if standard_errors == 'region'
        else None
    )
    design = regression.Design(np.column_stack([np.ones(n_genes), w]), x, standard_errors, clusters)

    in_universe = (
        scores
        .join(universe.select('geneId', 'geneIndex'), on='geneId')
        .filter(pl.col('score') > 0)
        .sort('unitId', 'geneIndex')
    )
    units = in_universe.group_by('unitId').agg(pl.len().alias('nGenesScored')).sort('unitId')
    unit_index = units.with_row_index('unitIndex').select('unitId', 'unitIndex')
    cells = in_universe.join(unit_index, on='unitId')
    logger.info(f'{units.height} units with at least one scored gene')

    frames = []
    for start in range(0, units.height, _UNIT_BATCH):
        stop = min(start + _UNIT_BATCH, units.height)
        batch = cells.filter(pl.col('unitIndex').is_between(start, stop - 1))
        y = np.zeros((n_genes, stop - start))
        y[batch['geneIndex'].to_numpy(), batch['unitIndex'].to_numpy() - start] = batch['score'].to_numpy()
        fit = design.fit(y)
        beta, se = fit.beta, fit.standard_error
        with np.errstate(divide='ignore', invalid='ignore'):
            z = beta / se
        z = np.where(np.isfinite(z), z, np.nan)
        # one-sided: only enrichment is of interest; corrections are across each unit's biosamples
        p = regression.upper_tail_p(z)
        frames.append(
            pl.DataFrame({
                'unitIndex': np.repeat(np.arange(start, stop), len(biosamples)),
                'biosampleId': np.tile(biosamples, stop - start),
                'beta': beta.ravel(),
                'standardError': se.ravel(),
                'zScore': z.ravel(),
                'pValue': p.ravel(),
                'pValueBonferroni': regression.bonferroni(p).ravel(),
                'qValue': regression.benjamini_hochberg(p).ravel(),
            }).with_columns(pl.col('unitIndex').cast(pl.UInt32))
        )
    if not frames:
        return pl.DataFrame(schema={'unitId': pl.String, 'nGenesScored': pl.UInt32, **_STATISTICS_SCHEMA})
    return (
        pl
        .concat(frames)
        .join(unit_index, on='unitIndex')
        .join(units, on='unitId')
        .drop('unitIndex')
        # undefined statistics (a biosample with no specificity variance left) are null, not NaN
        .with_columns(pl.col(pl.Float64).fill_nan(None))
    )


def biosample_enrichment(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate the biosample enrichment dataset.

    Args:
        source: paths keyed by l2g_prediction, credible_set, study, target and baseline_expression.
        destination: path of the biosample enrichment dataset.
        settings: `panels`, a list of `{datasourceId, biosampleType}`; and optionally
            `pool_by_disease` (default false), `standard_errors` (default `region`),
            `region_window` and `density_window` (both 1 Mb by default).
        config: unused.

    Raises:
        ValueError: if `standard_errors` is not one of the supported kinds.
    """
    panels = [Panel.from_settings(entry) for entry in settings['panels']]
    pool_by_disease = bool(settings.get('pool_by_disease', False))
    standard_errors = settings.get('standard_errors', 'region')
    if standard_errors not in {'region', 'gene', 'classical'}:
        raise ValueError(f'unsupported standard_errors {standard_errors!r}')
    region_window = int(settings.get('region_window', 1_000_000))
    density_window = int(settings.get('density_window', 1_000_000))

    covariates = gene_covariates(scan_dataset(source['target']), density_window)
    units = gwas_studies(scan_dataset(source['study']), pool_by_disease).collect()
    study_scores = study_gene_scores(
        scan_dataset(source['l2g_prediction']), scan_dataset(source['credible_set']), units.lazy()
    ).collect()
    # when pooling, a gene's score for the group is its highest over the group's studies
    scores = study_scores.group_by('unitId', 'geneId').agg(pl.col('score').max())
    # only studies with predictions of their own get the group's result; one with no credible
    # sets contributed nothing to it
    members = study_scores.select('studyId', 'unitId').unique()
    pooled = members.group_by('unitId').agg(pl.len().cast(pl.Int32).alias('nStudiesPooled'))
    logger.info(f'{members.height} studies in {pooled.height} units')

    baseline_expression = scan_dataset(source['baseline_expression'])
    results = []
    for panel in panels:
        logger.info(f'scoring panel {panel}')
        fitted = score_panel(
            scores,
            specificity_matrix(baseline_expression, panel),
            covariates,
            standard_errors=standard_errors,
            region_window=region_window,
        )
        results.append(
            fitted
            .join(members, on='unitId')
            .join(pooled, on='unitId')
            .with_columns(
                pl.lit(panel.datasource_id).alias('datasourceId'),
                pl.lit(panel.biosample_type).alias('biosampleType'),
            )
        )
    output = (
        pl
        .concat(results, how='diagonal_relaxed')
        .select([pl.col(name).cast(dtype) for name, dtype in OUTPUT_SCHEMA.items()])
        .sort('studyId', 'datasourceId', 'biosampleType', 'biosampleId')
    )
    write_dataset(output, destination)
    logger.info(f'wrote {output.height} rows')
