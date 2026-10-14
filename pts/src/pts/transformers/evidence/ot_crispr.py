"""OT CRISPR (ot_crispr) evidence, from OTAR genome-wide CRISPR/Cas9 knock-out screens.

- The results arrive in MAGeCK format.
- The study level metadata comes from a spreadsheet, versioned in the PPP evidence configuration repository.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import reduce
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.run import run_evidence
from pts.transformers.utils.dataset import scan_dataset

OT_CRISPR = EvidenceDatasource(
    datasource_id='ot_crispr',
    # order and repetition feed the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'studyId',
        'diseaseFromSourceMappedId',
        'targetFromSourceId',
        'resourceScore',
        'statisticalTestTail',
    ],
    score=pl.lit(1.0),
)
#: Java's `\s`, which is ASCII-only.
_LEADING_WHITESPACE = r'^[ \t\n\x0B\f\r]+'


@dataclass
class DataCheck:
    """A single quality check to apply to the study table.

    Attributes:
        condition: the condition failing rows meet.
        name: human-readable name for the check.
        action: action to take on failing rows; only 'drop' is supported.
        report_fn: extracts a summary from the failing rows.
        message_fn: formats the summary into a log message.
    """

    condition: pl.Expr
    name: str
    action: str
    report_fn: Callable[[pl.DataFrame], Any]
    message_fn: Callable[[Any], str]


def _study_ids(frame: pl.DataFrame) -> str:
    return ', '.join(frame['studyId'].drop_nulls().unique(maintain_order=True).to_list())


STUDY_CHECKS = [
    DataCheck(
        name='StudyId assertion',
        condition=pl.col('studyId').is_null(),
        action='drop',
        report_fn=lambda df: df.height,
        message_fn=lambda result: f'Rows with missing studyId: {result}',
    ),
    DataCheck(
        name='Disease id presence assertion',
        condition=pl.col('diseases').is_null(),
        action='drop',
        report_fn=_study_ids,
        message_fn=lambda result: f'Studies with missing diseaseId: {result}',
    ),
    DataCheck(
        name='DataFile presence assertion',
        condition=pl.col('dataFile').is_null(),
        action='drop',
        report_fn=_study_ids,
        message_fn=lambda result: f'Studies with missing dataset: {result}',
    ),
    DataCheck(
        name='Proper project id assertion',
        condition=pl.col('projectId').is_null() | ~pl.col('projectId').str.starts_with('OTAR'),
        action='drop',
        report_fn=_study_ids,
        message_fn=lambda result: f'Studies with invalid projectId: {result}',
    ),
]


def apply_check(study_table: pl.DataFrame, check: DataCheck) -> pl.DataFrame:
    """Apply a single data quality check, logging a warning and dropping failing rows if needed."""
    if result := check.report_fn(study_table.filter(check.condition)):
        logger.warning(check.message_fn(result))
        if check.action == 'drop':
            study_table = study_table.filter(~check.condition)
    return study_table


def split_column_value(column: pl.Expr, separator: str = '|') -> pl.Expr:
    """Remove leading whitespace and split by `separator`, keeping the first of each repeated value."""
    return column.str.replace(_LEADING_WHITESPACE, '').str.split(separator).list.unique(maintain_order=True)


def process_studies(study_table: pl.DataFrame) -> pl.DataFrame:
    """Parse the study table: drop invalid rows, split multi-valued fields and collect replicates.

    Args:
        study_table: the raw study table, every column a string.

    Returns:
        One row per study and parameter set, with its replicates as a list.
    """
    keys = [
        'studyId',
        'projectId',
        'projectDescription',
        'studyOverview',
        'releaseVersion',
        'releaseDate',
        'diseaseFromSourceMappedId',
        'isCellTypeDerived',
        'crisprScreenLibrary',
        'crisprStudyMode',
        'geneticBackground',
        'cellType',
        'cellLineBackground',
        'contrast',
        'filterColumns',
        'threshold',
    ]
    return (
        reduce(apply_check, STUDY_CHECKS, study_table)
        .filter(pl.col('projectId').str.starts_with('OTAR'))
        .with_columns(
            pl.col('releaseDate').str.strip_chars().str.to_date('%Y-%m-%d', strict=False),
            split_column_value(pl.col('diseases')).alias('diseaseFromSourceMappedId'),
            split_column_value(pl.col('filterColumn'), ',').alias('filterColumns'),
            pl.col('threshold').str.strip_chars().cast(pl.Float64, strict=False),
            pl.col('replicateNumber').fill_null('1').alias('replicateId'),
        )
        .group_by(keys, maintain_order=True)
        .agg(pl.struct('dataFile', 'ControlDataset', 'replicateId').alias('replicates'))
    )


def adjust_column_names(raw_data: pl.DataFrame) -> pl.DataFrame:
    """Normalise MAGeCK labels such as `pos.p-value` to `pos|p-value`, by the separator of the fourth column.

    Raises:
        ValueError: if the fourth column has neither `|` nor `.`.
    """
    label = raw_data.columns[3]
    if '|' in label:
        separator = '|'
    elif '.' in label:
        separator = '.'
    else:
        raise ValueError(f'Unrecognized label separator in {label}')
    return raw_data.rename({c: c.replace(separator, '|') for c in raw_data.columns})


def read_and_filter_mageck_file(mageck_file: str, filter_columns: list[str], threshold: float) -> pl.DataFrame:
    """Read a MAGeCK file and keep the genes whose smallest filter column value is below `threshold`.

    Args:
        mageck_file: location of the MAGeCK gene summary.
        filter_columns: the columns to filter on.
        threshold: the significance threshold.

    Returns:
        The hits, with the tail of the distribution their smallest value came from.
    """
    logger.info(f'reading MAGeCK file: {mageck_file}')
    raw_data = adjust_column_names(scan_dataset(mageck_file, format='tsv', infer_schema=False).collect())
    values = {c: pl.col(c).cast(pl.Float64, strict=False) for c in filter_columns}
    score = pl.col('resourceScore')
    source_label = pl.coalesce([pl.when(value == score).then(pl.lit(c)) for c, value in values.items()])
    is_pos, is_neg = source_label.str.contains('pos', literal=True), source_label.str.contains('neg', literal=True)
    return (
        raw_data
        .with_columns(pl.min_horizontal(values.values()).alias('resourceScore'))
        .filter(score < pl.lit(threshold, dtype=pl.Float64))
        .select(
            pl.col('id').str.split('_').list.first().alias('targetFromSourceId'),
            pl
            .when(is_pos)
            .then(pl.col('pos|lfc'))
            .when(is_neg)
            .then(pl.col('neg|lfc'))
            .cast(pl.Float64, strict=False)
            .cast(pl.Float32)
            .alias('log2FoldChangeValue'),
            pl
            .when(is_pos)
            .then(pl.lit('upper tail'))
            .when(is_neg)
            .then(pl.lit('lower tail'))
            .alias('statisticalTestTail'),
            'resourceScore',
        )
    )


def process_replicate(
    data_file: str,
    control_dataset: str | None,
    filter_columns: list[str],
    threshold: float,
) -> pl.DataFrame:
    """Find the hits of one replicate, excluding those that are also hits in its control, if any."""
    hits = read_and_filter_mageck_file(data_file, filter_columns, threshold)
    if control_dataset:
        control = read_and_filter_mageck_file(control_dataset, filter_columns, threshold)
        hits = hits.join(control.select('targetFromSourceId').unique(), on='targetFromSourceId', how='anti')
    return hits


def process_study(study: dict[str, Any], data_path: str) -> pl.DataFrame:
    """Find the hits of one study: genes that are hits in every replicate, with the best replicate's values.

    A gene counts once per replicate, however many of the replicate's ids it has. The best replicate
    is the one with the smallest `resourceScore`, the first on a tie.

    Args:
        study: one row of `process_studies`.
        data_path: base directory of the per-project MAGeCK files.

    Returns:
        One row per hit gene.
    """
    project = f'{data_path}/{study["projectId"]}'
    replicates = [
        process_replicate(
            data_file=f'{project}/{replicate["dataFile"]}',
            control_dataset=f'{project}/{replicate["ControlDataset"]}' if replicate['ControlDataset'] else None,
            filter_columns=study['filterColumns'],
            threshold=study['threshold'],
        )
        for replicate in study['replicates']
    ]
    return (
        pl
        .concat([hits.with_columns(replicate=pl.lit(i)) for i, hits in enumerate(replicates)])
        .group_by('targetFromSourceId', maintain_order=True)
        .agg(
            pl.col('replicate').n_unique().alias('found'),
            pl
            .col('log2FoldChangeValue', 'resourceScore', 'statisticalTestTail')
            .get(pl.col('resourceScore').arg_min()),
        )
        .filter(pl.col('found') == len(replicates))
        .select(
            'targetFromSourceId',
            'log2FoldChangeValue',
            'resourceScore',
            'statisticalTestTail',
            pl.lit(study['studyId'], dtype=pl.String).alias('studyId'),
        )
    )


def generate(study_table: str, ot_crispr_data: str) -> pl.LazyFrame:
    """Build OT CRISPR evidence from the study table and the per-project MAGeCK files.

    Args:
        study_table: location of the study table csv.
        ot_crispr_data: base directory of the per-project MAGeCK files.

    Returns:
        Raw OT CRISPR evidence, one row per hit gene, study and mapped disease.
    """
    studies = process_studies(scan_dataset(study_table, format='csv', infer_schema=False).collect())
    hits = pl.concat([process_study(study, ot_crispr_data) for study in studies.iter_rows(named=True)])
    return (
        hits
        .join(studies, on='studyId', how='inner', maintain_order='left')
        .select(
            'projectId',
            'projectDescription',
            'releaseDate',
            'releaseVersion',
            'studyId',
            'studyOverview',
            'contrast',
            'crisprScreenLibrary',
            'cellType',
            'cellLineBackground',
            'diseaseFromSourceMappedId',
            'targetFromSourceId',
            'log2FoldChangeValue',
            'resourceScore',
            'statisticalTestTail',
            pl.lit('ot_crispr').alias('datasourceId'),
            pl.lit('ot_partner').alias('datatypeId'),
        )
        .explode('diseaseFromSourceMappedId', empty_as_null=False, keep_nulls=False)
        .lazy()
    )


def ot_crispr(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process OT CRISPR evidence.

    Args:
        source: `study_table`, `ot_crispr_data`, `disease` and `target` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: unused.
        config: unused.
    """
    run_evidence(
        generate(source['study_table'], source['ot_crispr_data']),
        OT_CRISPR,
        disease=source['disease'],
        target=source['target'],
        destination=destination,
    )
