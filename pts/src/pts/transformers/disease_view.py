"""Disease view dataset generation.

Every disease row in the served column order, with its synonyms as relation/terms pairs and the
studies mapped to it and to its descendants.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from ot_datamodels.outputs.disease import Synonyms
from ot_datamodels.views.disease_view import DiseaseView
from otter.config.model import Config

from pts.schemas.dataset.conform import conform
from pts.task_outputs import writes
from pts.transformers.utils.dataset import scan_dataset, write_dataset

_EMPTY = pl.lit([], dtype=pl.List(pl.String))


def _synonyms() -> pl.Expr:
    """`synonyms` as one `{relation, terms}` per `Synonyms` field, in field order; null terms are `[]`."""
    return pl.concat_list([
        pl.struct(
            relation=pl.lit(relation),
            terms=pl.col('synonyms').struct.field(relation).fill_null(_EMPTY),
        )
        for relation in Synonyms.model_fields
    ]).alias('synonyms')


def disease_view_frame(disease: pl.DataFrame, study: pl.DataFrame) -> pl.DataFrame:
    """Every disease row, in its order, in `DiseaseView` columns.

    `studyIds`: the studies naming the disease. `indirectStudyIds`: the studies naming any of its
    descendants. Both distinct, ascending, `[]` for none; empty study ids and null disease ids are
    ignored, and study disease ids absent from `disease` are dropped.
    """
    direct = (
        study
        .select('studyId', diseaseId=pl.col('diseaseIds'))
        .explode('diseaseId')
        .filter(pl.col('diseaseId').is_not_null() & (pl.col('studyId') != ''))
        .unique()
        .group_by('diseaseId')
        .agg(studyIds=pl.col('studyId').sort())
    )
    indirect = (
        disease
        .select('id', diseaseId=pl.col('descendants'))
        .explode('diseaseId')
        .join(direct, on='diseaseId')
        .select('id', indirectStudyIds=pl.col('studyIds'))
        .explode('indirectStudyIds')
        .group_by('id')
        .agg(pl.col('indirectStudyIds').unique().sort())
    )
    return (
        disease
        .join(direct.rename({'diseaseId': 'id'}), on='id', how='left', maintain_order='left')
        .join(indirect, on='id', how='left', maintain_order='left')
        .rechunk()
        .with_columns(
            _synonyms(),
            pl.col('studyIds', 'indirectStudyIds').fill_null(_EMPTY),
            isTherapeuticArea=pl.col('ontology').struct.field('isTherapeuticArea'),
        )
        .select(list(DiseaseView.model_fields))
    )


@writes(view=DiseaseView)
def disease_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the disease view and write it.

    Args:
        source: `disease` and `study` locations.
        destination: `view`, the disease view directory.
        settings: unused.
        config: unused.
    """
    logger.info('load disease view inputs')
    disease, study = pl.collect_all([
        scan_dataset(source['disease']),
        scan_dataset(source['study']).select('studyId', 'diseaseIds'),
    ])
    result = disease_view_frame(disease, study)
    logger.info(f'{result.height} diseases in the disease view')
    write_dataset(conform(result, DiseaseView), str(destination['view']))
