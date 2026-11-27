"""Load precomputed CELLECT-LDSC results into the biosample enrichment schema.

CELLECT-LDSC writes one `<panel>__<studyId>.cell_type_results.txt` per GWAS study and specificity
panel, where `<panel>` is `<datasourceId>-<biosampleType>` (`gtex-tissue`,
`tabula_sapiens-tissue_celltype`, ...). Each row is one annotation, `<panel>__<biosampleId>` or, for
a tissue x cell type panel, `<panel>__<cell type id>__<tissue id>`. Biosamples are labelled from
baseline expression, as `biosample_enrichment` labels the L2G regression's.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.outputs.biosample_enrichment import BiosampleEnrichment
from otter.config.model import Config

from pts.schemas.biosample_enrichment import biosample_enrichment_schema
from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.biosample_enrichment import PAIR_SEPARATOR, Panel, biosample_sources
from pts.transformers.utils.dataset import list_parts, scan_dataset, write_dataset

ENRICHMENT_APPROACH = 'cellect_ldsc'
"""Value of `enrichmentApproach` for this method."""

RESULT_GLOB = '*.cell_type_results.txt'
"""The final results; CELLECT's `*.cell_type_results.tmp.txt` and logs are not read."""

RESULT_SCHEMA = pl.Schema({
    'Name': pl.String,
    'Coefficient': pl.Float64,
    'Coefficient_std_error': pl.Float64,
    'Coefficient_P_value': pl.Float64,
})

_FILE_NAME = r'([^/]+?)__([^/]+)\.cell_type_results\.txt$'

_PAIRED = pl.col('biosampleType') == 'tissue_celltype'
"""Whether a row is of a tissue x cell type panel, whose annotations name two biosamples."""


def read_results(path: str) -> pl.DataFrame:
    """Read every result file below `path`, parsing its panel, study and biosample key.

    Args:
        path: directory of CELLECT-LDSC `*.cell_type_results.txt` files.

    Returns:
        `datasourceId`, `biosampleType`, `studyId`, `biosampleKey` (the biosample's key in
        `biosample_sources`), `beta`, `betaError` and `pValue`, one row per study and annotation.

    Raises:
        ValueError: for a file or annotation name that does not follow the CELLECT layout.
    """
    raw = pl.scan_csv(
        list_parts(path, RESULT_GLOB), separator='\t', schema=RESULT_SCHEMA, include_file_paths='file'
    ).collect()
    names = pl.col('file').str.extract_groups(_FILE_NAME)
    parsed = (
        raw
        .with_columns(names.struct.field('1').alias('panel'), names.struct.field('2').alias('studyId'))
        .with_columns(
            pl.col('panel').str.split_exact('-', 1).struct.rename_fields(['datasourceId', 'biosampleType']),
            pl.col('Name').str.strip_prefix(pl.concat_str('panel', pl.lit('__'))).str.split('__').alias('ids'),
        )
        .unnest('panel')
    )
    valid = (
        pl.col('studyId').is_not_null()
        & pl.col('Name').str.starts_with(pl.concat_str('datasourceId', pl.lit('-'), 'biosampleType', pl.lit('__')))
        & (pl.col('ids').list.len() == pl.when(_PAIRED).then(2).otherwise(1))
    )
    invalid = parsed.filter(~valid.fill_null(False))
    if not invalid.is_empty():
        raise ValueError(f'unexpected CELLECT-LDSC result names, e.g. {invalid.select("file", "Name").head(5).rows()}')
    return parsed.select(
        'datasourceId',
        'biosampleType',
        'studyId',
        pl
        .when(_PAIRED)
        .then(pl.concat_str(pl.col('ids').list.get(1), pl.col('ids').list.get(0), separator=PAIR_SEPARATOR))
        .otherwise(pl.col('ids').list.first())
        .alias('biosampleKey'),
        pl.col('Coefficient').alias('beta'),
        pl.col('Coefficient_std_error').alias('betaError'),
        pl.col('Coefficient_P_value').alias('pValue'),
    )


@writes(BiosampleEnrichment)
def load_cellect_ldsc(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Reshape precomputed CELLECT-LDSC results into the biosample enrichment schema, one row per study.

    Args:
        source: paths keyed by `cellect_ldsc` (the directory of result files) and `baseline_expression`.
        destination: path of the reshaped results.
        settings: unused.
        config: unused.
    """
    results = read_results(source['cellect_ldsc'])
    baseline_expression = scan_dataset(source['baseline_expression'])
    labelled = []
    for (datasource_id, biosample_type), rows in results.group_by('datasourceId', 'biosampleType', maintain_order=True):
        panel = Panel.from_settings({'datasourceId': datasource_id, 'biosampleType': biosample_type})
        panel_rows = rows.join(biosample_sources(baseline_expression, panel), on='biosampleKey', how='left')
        unlabelled = panel_rows.filter(pl.col('biosampleId').is_null())['biosampleKey'].unique().sort()
        if len(unlabelled):
            logger.warning(
                f'{len(unlabelled)} biosamples of panel {panel} are not in baseline expression: {unlabelled.to_list()}'
            )
        labelled.append(
            panel_rows.with_columns(
                # a biosample missing from baseline expression keeps the id CELLECT names it by
                pl.col('biosampleId').fill_null(
                    pl
                    .when(_PAIRED)
                    .then(pl.col('biosampleKey').str.split(PAIR_SEPARATOR).list.last())
                    .otherwise(pl.col('biosampleKey'))
                ),
                pl.col('tissueBiosampleId').fill_null(
                    pl.when(_PAIRED).then(pl.col('biosampleKey').str.split(PAIR_SEPARATOR).list.first())
                ),
            )
        )
    output = (
        pl
        .concat(labelled, how='diagonal_relaxed')
        .with_columns(
            pl.lit(None, pl.String).alias('diseaseId'),
            pl.col('datasourceId').alias('biosampleDataset'),
            pl.lit(ENRICHMENT_APPROACH).alias('enrichmentApproach'),
            pl.col('beta', 'betaError', 'pValue').fill_nan(None),
        )
        .select([pl.col(name).cast(dtype) for name, dtype in biosample_enrichment_schema.items()])
        .sort('studyId', 'biosampleDataset', 'biosampleId', 'tissueBiosampleId', nulls_last=True)
    )
    logger.info(f'{output.height} rows over {output["studyId"].n_unique()} studies')
    write_dataset(conform(output, BiosampleEnrichment), destination)
