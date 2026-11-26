"""Study view: every study with the ids of its credible sets, the served `study_view` (was `studies.sql`)."""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.study_view import StudyView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset

VIEW_SCHEMA = to_polars_schema(StudyView)


def build_study_view(study: pl.DataFrame, credible_set: pl.DataFrame) -> pl.DataFrame:
    """Every study row, in study order, then its credible-set ids, sorted, `[]` for none; conformed to `StudyView`.

    Args:
        study: the study output.
        credible_set: `id`, `studyId` of the credible set output.
    """
    ids = credible_set.group_by('studyId').agg(credibleSetIds=pl.col('id').sort())
    view = study.join(ids, on='studyId', how='left', maintain_order='left').with_columns(
        credibleSetIds=pl.col('credibleSetIds').fill_null(pl.lit([], dtype=VIEW_SCHEMA['credibleSetIds']))
    )
    return conform(view.select(VIEW_SCHEMA.names()), StudyView)


@writes(view=StudyView)
def study_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the study view and write it.

    Args:
        source: `study` and `credible_set` locations.
        destination: the `view` directory.
        settings: unused.
        config: unused.
    """
    study, credible_set = pl.collect_all([
        scan_dataset(source['study']),
        scan_dataset(source['credible_set']).select('id', 'studyId'),
    ])
    view = build_study_view(study, credible_set)
    logger.info(f'{view.height} studies, {credible_set.height} credible sets in the study view')
    write_dataset(view, str(destination['view']))
