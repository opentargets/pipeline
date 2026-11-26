"""L2G prediction by credible set: the served `l2g_prediction_by_credible_set_view` shape.

One row per credible set with its predictions by `score` descending, then `geneId`.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.views.l2g_prediction_by_credible_set_view import L2gPredictionByCredibleSetView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset

#: The served tuple: the output row.
FIELDS = ['credibleSetId', 'geneId', 'score', 'features', 'shapBaseValue']


def l2g_prediction_view_frame(predictions: pl.DataFrame) -> pl.DataFrame:
    """Every credible set of `l2g_prediction` with its predictions; rows by credible set."""
    return (
        predictions
        .rechunk()
        .sort(['credibleSetId', 'score', 'geneId'], descending=[False, True, False])
        .select('credibleSetId', row=pl.struct(FIELDS))
        .group_by('credibleSetId', maintain_order=True)
        .agg(l2g_predictions=pl.col('row'))
    )


@writes(view=L2gPredictionByCredibleSetView)
def l2g_prediction_view(
    source: str,
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the L2G prediction by credible set view and write it.

    Args:
        source: the `l2g_prediction` output.
        destination: `view`, the view directory.
        settings: unused.
        config: unused.
    """
    result = l2g_prediction_view_frame(scan_dataset(str(source)).select(FIELDS).collect())
    logger.info(f'{result.height} credible sets in the l2g prediction view')
    write_dataset(conform(result, L2gPredictionByCredibleSetView), str(destination['view']))
