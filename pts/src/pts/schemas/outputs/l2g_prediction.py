"""Locus-to-Gene (L2G) prediction: the `l2g_prediction` output.

Generated from the 26.09 release output (3,312,794 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, MD5_HEX


class Features(StructModel):
    """`features[]`."""

    name: str = Field(min_length=1, description='Name of the feature used in Locus-to-Gene (L2G) prediction')
    value: Annotated[float, PolarsDtype(pl.Float32)] = Field(ge=0.0, le=130.0, description='Value of the feature')
    shapValue: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=-0.15446002781391144, le=0.3564266860485077, description='SHAP value of the feature'
    )


class L2gPrediction(DatasetModel):
    """Locus-to-Gene (L2G) prediction."""

    dataset_name: ClassVar[str] = 'l2g_prediction'
    dataset_title: ClassVar[str] = 'Locus-to-Gene (L2G) prediction'
    dataset_description: ClassVar[str] = (
        'Predictions from our Locus-to-Gene (L2G) gene assignment model. The '
        'dataset contains all predictions for every combination of credible '
        'set and genes in the region as well as statistics to explain the '
        'model interpretation of the predictions'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'studyLocusId': '013d8db926243ead73230279a498d4f1',
        'geneId': 'ENSG00000069424',
        'score': 0.4182942807674408,
        'features': [{'name': 'eQtlColocClppMaximum', 'value': 0.0, 'shapValue': -0.0013690858613699675}],
        'shapBaseValue': 0.05781528353691101,
    }

    studyLocusId: Annotated[str, Reference('credible_set', 'studyLocusId'), PrimaryKey()] = Field(
        pattern=MD5_HEX, description='Study-locus identifier for the credible set'
    )
    geneId: Annotated[str, Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Identifier of the gene/target'
    )
    score: float = Field(
        ge=0.050000015646219254,
        le=0.9978274703025818,
        description=('Locus-to-Gene (L2G) score prediction for the given studyLocusId-geneId pair'),
    )
    features: list[Features] = Field(min_length=1, description='List of Locus-to-Gene (L2G) feature values')
    shapBaseValue: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=0.024695513769984245, le=0.14552989602088928, description='SHAP base value'
    )
