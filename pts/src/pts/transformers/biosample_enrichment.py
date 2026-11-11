"""Biosample enrichment dataset generation.

For each GWAS study and each tissue or cell type in a specificity panel, tests whether the genes
L2G prioritises are more specific to that biosample than gene length and gene density explain.
The regression itself is in `pts.transformers.enrichment.regression`.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Literal, get_args

import numpy as np
import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.biosample_enrichment import biosample_enrichment_schema, biosample_enrichment_statistics_schema
from pts.transformers.enrichment import regression
from pts.transformers.enrichment.regression import StandardErrors
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.target import tss_lut

BiosampleType = Literal['tissue', 'celltype', 'tissue_celltype']

_BIOSAMPLE_TYPES: tuple[BiosampleType, ...] = ('tissue', 'celltype', 'tissue_celltype')

_PAIR_SEPARATOR = '|'
"""Joins a pair's tissue and cell type ids into one column name of the specificity matrix."""


@dataclass(frozen=True)
class _PanelColumns:
    """How a panel's biosamples are read from the baseline expression dataset.

    `key` names a biosample within the panel: a column of the specificity matrix. `biosample` and
    `tissue` are the released `biosampleId` and `tissueBiosampleId`.
    """

    rows: pl.Expr
    key: pl.Expr
    biosample: pl.Expr
    tissue: pl.Expr
    source: pl.Expr


def _panel_columns(biosample_type: BiosampleType) -> _PanelColumns:
    tissue, celltype = pl.col('tissueBiosampleId'), pl.col('celltypeBiosampleId')
    no_tissue = pl.lit(None, pl.String)
    if biosample_type == 'tissue':
        return _PanelColumns(
            tissue.is_not_null() & celltype.is_null(), tissue, tissue, no_tissue, pl.col('tissueBiosampleFromSource')
        )
    if biosample_type == 'celltype':
        return _PanelColumns(
            celltype.is_not_null() & tissue.is_null(),
            celltype,
            celltype,
            no_tissue,
            pl.col('celltypeBiosampleFromSource'),
        )
    # a cell type profiled in one tissue: reported as the cell type, with the tissue alongside
    return _PanelColumns(
        tissue.is_not_null() & celltype.is_not_null(),
        pl.concat_str(tissue, celltype, separator=_PAIR_SEPARATOR),
        celltype,
        tissue,
        pl.concat_str(pl.col('tissueBiosampleFromSource'), pl.col('celltypeBiosampleFromSource'), separator=' | '),
    )


ENRICHMENT_APPROACH = 'l2g_regression'
"""Value of `enrichmentApproach` for this method."""


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
            ValueError: if `biosampleType` is not `tissue`, `celltype` or `tissue_celltype`.
        """
        kind = entry['biosampleType']
        if kind not in _BIOSAMPLE_TYPES:
            raise ValueError(f'biosampleType must be one of {list(_BIOSAMPLE_TYPES)}, got {kind!r}')
        return cls(datasource_id=entry['datasourceId'], biosample_type=kind)


@dataclass(frozen=True)
class Settings:
    """The step's settings, with their defaults."""

    panels: tuple[Panel, ...]
    pool_by_disease: bool = False
    """Score the studies of one disease together, each gene at its highest L2G among them."""
    standard_errors: StandardErrors = 'region'
    region_window: int = 1_000_000
    """Width, in base pairs, of the fixed bins `chromosome:floor(tss / region_window)` that
    `region` standard errors cluster genes by."""
    density_window: int = 1_000_000
    """Half-width, in base pairs, of the window around each TSS that gene density counts TSSs in."""

    @classmethod
    def from_settings(cls, settings: dict[str, Any]) -> Settings:
        """Read the settings from the step's config.

        Args:
            settings: `panels`, a list of `{datasourceId, biosampleType}`, and any of the other fields.

        Returns:
            The settings.

        Raises:
            ValueError: for an unknown key or `standard_errors` kind, or a `region_window` given
                with standard errors that do not use it.
        """
        unknown = set(settings) - {f.name for f in fields(cls)}
        if unknown:
            raise ValueError(f'unknown settings {sorted(unknown)}')
        standard_errors = settings.get('standard_errors', cls.standard_errors)
        if standard_errors not in get_args(StandardErrors):
            raise ValueError(f'unsupported standard_errors {standard_errors!r}')
        if 'region_window' in settings and standard_errors != 'region':
            raise ValueError(f'region_window is only used by region standard errors, not {standard_errors!r}')
        return cls(
            panels=tuple(Panel.from_settings(entry) for entry in settings['panels']),
            pool_by_disease=bool(settings.get('pool_by_disease', cls.pool_by_disease)),
            standard_errors=standard_errors,
            region_window=int(settings.get('region_window', cls.region_window)),
            density_window=int(settings.get('density_window', cls.density_window)),
        )


def gene_covariates(target: pl.LazyFrame, density_window: int) -> pl.DataFrame:
    """Gene length, TSS density and position per gene, for every gene with a usable location.

    Long genes in gene-poor regions are more often both GWAS hits and tissue specific, so these
    are adjusted for to stop that alone looking like enrichment.

    The TSS is the target's own or, where it has none, the strand-aware end of its location
    (`tss_lut`).

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
            pl.col('genomicLocation').struct.field('chromosome'),
            (pl.col('genomicLocation').struct.field('end') - pl.col('genomicLocation').struct.field('start'))
            .abs()
            .alias('length'),
        )
        .join(tss_lut(target), on='geneId', how='left')
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


def gwas_studies(study: pl.LazyFrame, pool_by_disease: bool) -> pl.DataFrame:
    """Assign every GWAS study to the unit it is scored in, and label the unit as it is released.

    A unit is the study itself or, when pooling, every study of the same single disease.
    Studies with no disease or with several diseases are not pooled.

    Args:
        study: the study index.
        pool_by_disease: pool studies that have the same single disease into one unit.

    Returns:
        `studyId`, the unit's `unitId`, and its released `unitDiseaseId` and `unitStudyId`, one row
        per GWAS study. Exactly one of the two labels is set: the disease of a pooled unit, or the
        study of a unit of one.
    """
    diseases = pl.col('diseaseIds').list.unique()
    pooled = pl.when(diseases.list.len() == 1).then(diseases.list.first()) if pool_by_disease else pl.lit(None)
    members = (
        study
        .filter(pl.col('studyType') == 'gwas')
        .select('studyId', pooled.cast(pl.String).alias('unitDiseaseId'))
        .with_columns(pl.when(pl.col('unitDiseaseId').is_null()).then(pl.col('studyId')).alias('unitStudyId'))
        .collect()
    )
    units = (
        members
        .select('unitDiseaseId', 'unitStudyId')
        .unique()
        .sort('unitDiseaseId', 'unitStudyId', nulls_last=True)
        .with_row_index('unitId')
    )
    return members.join(units, on=['unitDiseaseId', 'unitStudyId'], nulls_equal=True).select(
        'studyId', 'unitId', 'unitDiseaseId', 'unitStudyId'
    )


def study_gene_scores(l2g: pl.LazyFrame, credible_set: pl.LazyFrame, units: pl.LazyFrame) -> pl.LazyFrame:
    """Highest L2G score per GWAS study and gene, over the study's credible sets.

    The release keeps only scores above its L2G threshold, so a missing gene is read as 0 later.

    Args:
        l2g: the L2G predictions.
        credible_set: the credible sets, for the study of each locus.
        units: `studyId` and `unitId`, from `gwas_studies`; studies not in it are dropped.

    Returns:
        `studyId`, `unitId`, `geneId` and `score`.
    """
    sets = credible_set.select(pl.col('id').alias('credibleSetId'), 'studyId').unique()
    return (
        l2g
        .select('credibleSetId', 'geneId', 'score')
        .join(sets, on='credibleSetId')
        .join(units.select('studyId', 'unitId'), on='studyId')
        .group_by('studyId', 'unitId', 'geneId')
        .agg(pl.col('score').max())
    )


def specificity_matrix(baseline_expression: pl.LazyFrame, panel: Panel) -> pl.DataFrame:
    """A panel's specificity scores, one row per gene and one column per biosample.

    Args:
        baseline_expression: the baseline expression dataset.
        panel: the panel to read.

    Returns:
        `geneId` and then one Float64 column per biosample, sorted. A column is named by the
        biosample id or, for a tissue x cell type panel, `<tissue id>|<cell type id>`. Missing
        values are null.

    Raises:
        ValueError: if the panel has no specificity scores.
    """
    columns = _panel_columns(panel.biosample_type)
    long = (
        baseline_expression
        .filter(
            pl.col('datasourceId') == panel.datasource_id,
            columns.rows,
            pl.col('specificity_score').is_not_null(),
        )
        # a gene and biosample repeat when several source ids map to one target, with equal scores
        .group_by(pl.col('targetId').alias('geneId'), columns.key.alias('biosampleId'))
        .agg(pl.col('specificity_score').max().cast(pl.Float64))
        .collect()
    )
    if long.is_empty():
        raise ValueError(f'no specificity scores for panel {panel}')
    wide = long.pivot(on='biosampleId', index='geneId', values='specificity_score')
    biosamples = sorted(c for c in wide.columns if c != 'geneId')
    return wide.select('geneId', *biosamples).sort('geneId')


def biosample_sources(baseline_expression: pl.LazyFrame, panel: Panel) -> pl.DataFrame:
    """The released ids and source name of each biosample in a panel.

    Args:
        baseline_expression: the baseline expression dataset.
        panel: the panel to read.

    Returns:
        `biosampleKey` (the biosample's column in `specificity_matrix`), `biosampleId`,
        `tissueBiosampleId` and `biosampleFromSource`, one row per biosample. When several source
        biosamples map to one id, their names are sorted and joined with `; `. A tissue x cell type
        pair's source name is `<tissue> | <cell type>`.
    """
    columns = _panel_columns(panel.biosample_type)
    sources = (
        baseline_expression
        .filter(pl.col('datasourceId') == panel.datasource_id, columns.rows)
        .select(
            columns.key.alias('biosampleKey'),
            columns.biosample.alias('biosampleId'),
            columns.tissue.alias('tissueBiosampleId'),
            columns.source.alias('source'),
        )
        .unique()
        .group_by('biosampleKey', 'biosampleId', 'tissueBiosampleId')
        .agg(pl.col('source').drop_nulls().sort().str.join('; ').alias('biosampleFromSource'))
        .with_columns(pl.col('biosampleFromSource').replace('', None))
        .collect()
    )
    shared = sources.filter(pl.col('biosampleFromSource').str.contains('; ', literal=True)).height
    if shared:
        logger.warning(f'{shared} biosamples in panel {panel} have several source biosamples')
    return sources


def score_panel(
    scores: pl.DataFrame,
    specificity: pl.DataFrame,
    covariates: pl.DataFrame,
    settings: Settings,
    *,
    max_bytes: int = regression.DEFAULT_MAX_BYTES,
) -> pl.DataFrame:
    """Fit every unit against every biosample of one panel.

    Args:
        scores: `unitId`, `geneId` and `score`, the highest per unit and gene.
        specificity: the panel, from `specificity_matrix`.
        covariates: per-gene covariates, from `gene_covariates`.
        settings: for the standard errors and their region window.
        max_bytes: passed to `regression.enrichment`.

    Returns:
        `unitId`, `biosampleId`, `beta`, `betaError` and `pValue`. A unit with no scored gene in
        the universe has no rows.
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
    logger.info(f'{universe.height} genes and {len(biosamples)} biosamples in the universe')

    cells = (
        scores
        .join(universe.select('geneId', 'geneIndex'), on='geneId')
        .filter(pl.col('score') > 0)
        .with_columns(pl.col('unitId').rank('dense').cast(pl.UInt32).sub(1).alias('unitIndex'))
    )
    unit_index = cells.select('unitId', 'unitIndex').unique()
    logger.info(f'{unit_index.height} units with at least one scored gene')

    results = regression.enrichment(
        universe.select('logLength', 'density').to_numpy().astype(np.float64),
        universe.select(biosamples).fill_null(0.0).to_numpy().astype(np.float64),
        regression.Outcomes(
            genes=cells['geneIndex'].to_numpy(),
            units=cells['unitIndex'].to_numpy(),
            values=cells['score'].to_numpy(),
            n_units=unit_index.height,
        ),
        settings.standard_errors,
        regression.region_labels(universe['chromosome'].to_numpy(), universe['tss'].to_numpy(), settings.region_window),
        max_bytes=max_bytes,
    )
    frames = [
        pl.DataFrame({
            'unitIndex': np.repeat(np.arange(batch.start, batch.start + len(batch.beta)), len(biosamples)),
            'biosampleId': np.tile(biosamples, len(batch.beta)),
            'beta': batch.beta.ravel(),
            'betaError': batch.standard_error.ravel(),
            'pValue': batch.p_value.ravel(),
        }).with_columns(pl.col('unitIndex').cast(pl.UInt32))
        for batch in results
    ]
    if not frames:
        return pl.DataFrame(schema={'unitId': pl.UInt32, **biosample_enrichment_statistics_schema})
    return (
        pl
        .concat(frames)
        .join(unit_index, on='unitIndex')
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
        settings: as `Settings.from_settings` reads them.
        config: unused.
    """
    options = Settings.from_settings(settings)
    covariates = gene_covariates(scan_dataset(source['target']), options.density_window)
    units = gwas_studies(scan_dataset(source['study']), options.pool_by_disease)
    study_scores = study_gene_scores(
        scan_dataset(source['l2g_prediction']), scan_dataset(source['credible_set']), units.lazy()
    ).collect()
    # when pooling, a gene's score for the disease is its highest over the disease's studies
    scores = study_scores.group_by('unitId', 'geneId').agg(pl.col('score').max())
    labels = units.select(
        'unitId', pl.col('unitDiseaseId').alias('diseaseId'), pl.col('unitStudyId').alias('studyId')
    ).unique('unitId')
    logger.info(
        f'{study_scores["studyId"].n_unique()} studies in {study_scores["unitId"].n_unique()} units with L2G scores'
    )

    baseline_expression = scan_dataset(source['baseline_expression'])
    results = []
    for panel in options.panels:
        logger.info(f'scoring panel {panel}')
        fitted = score_panel(scores, specificity_matrix(baseline_expression, panel), covariates, options)
        results.append(
            fitted
            .join(labels, on='unitId')
            # the fit names each biosample by its specificity matrix column
            .rename({'biosampleId': 'biosampleKey'})
            .join(biosample_sources(baseline_expression, panel), on='biosampleKey', how='left')
            .drop('biosampleKey')
            .with_columns(
                pl.lit(panel.datasource_id).alias('biosampleDataset'),
                pl.lit(ENRICHMENT_APPROACH).alias('enrichmentApproach'),
            )
        )
    output = (
        pl
        .concat(results, how='diagonal_relaxed')
        .select([pl.col(name).cast(dtype) for name, dtype in biosample_enrichment_schema.items()])
        .sort('diseaseId', 'studyId', 'biosampleDataset', 'biosampleId', 'tissueBiosampleId', nulls_last=True)
    )
    write_dataset(output, destination)
    logger.info(f'wrote {output.height} rows')
