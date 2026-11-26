"""L2G prediction by credible set: the `l2g_prediction_by_credible_set_view` view, served as `l2g_predictions`.

Written by pts `l2g_prediction_view`. One row per credible set with its `l2g_prediction` rows, by `score`
descending, then `geneId`.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.l2g_prediction import Features, L2gPrediction
from ot_datamodels.serving import Serving

CredibleSetId = from_output(L2gPrediction, 'credibleSetId', primary_key=True)
RowCredibleSetId = from_output(L2gPrediction, 'credibleSetId')
GeneId = from_output(L2gPrediction, 'geneId')
Score = from_output(L2gPrediction, 'score')
ShapBaseValue = from_output(L2gPrediction, 'shapBaseValue')
FeatureValue = from_output(Features, 'value')
FeatureShapValue = from_output(Features, 'shapValue')

_EXAMPLE = L2gPrediction.example


class L2gFeature(StructModel):
    """`l2g_predictions[].features[]`: an output feature, `name` without LowCardinality."""

    name: str = Field(min_length=1, description=Features.model_fields['name'].description)
    value: FeatureValue
    shapValue: FeatureShapValue


class L2gPredictionRow(StructModel):
    """`l2g_predictions[]`: an `l2g_prediction` row."""

    credibleSetId: RowCredibleSetId
    geneId: GeneId
    score: Score
    features: list[L2gFeature] = Field(description=L2gPrediction.model_fields['features'].description)
    shapBaseValue: ShapBaseValue


class L2gPredictionByCredibleSetView(DatasetModel):
    """L2G prediction by credible set."""

    dataset_name: ClassVar[str] = 'l2g_prediction_by_credible_set_view'
    dataset_title: ClassVar[str] = 'L2G prediction by credible set'
    dataset_description: ClassVar[str] = 'The Locus-to-Gene (L2G) predictions of each credible set, best first'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = {'credibleSetId': _EXAMPLE['credibleSetId'], 'l2g_predictions': [_EXAMPLE]}

    credibleSetId: CredibleSetId
    l2g_predictions: list[L2gPredictionRow] = Field(
        min_length=1, description='L2G predictions of the credible set, by score descending, then geneId'
    )
