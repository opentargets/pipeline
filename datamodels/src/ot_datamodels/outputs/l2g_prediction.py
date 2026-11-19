"""Locus-to-Gene (L2G) prediction: the `l2g_prediction` output.

Generated from the 26.12 release output (3,312,601 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, MD5_HEX


class Features(StructModel):
    """`features[]`."""

    name: str = Field(min_length=1, description='Name of the feature used in Locus-to-Gene (L2G) prediction')
    value: Annotated[float, PolarsDtype(pl.Float32)] = Field(ge=0.0, description='Value of the feature')
    shapValue: Annotated[float, PolarsDtype(pl.Float32)] = Field(description='SHAP value of the feature')


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
        'credibleSetId': '0000132125f56a4c99a6ef701eaa299d',
        'geneId': 'ENSG00000070961',
        'score': 0.5039282441139221,
        'features': [{'name': 'eQtlColocClppMaximum', 'value': 0.0, 'shapValue': -0.0010958482744172215}],
        'shapBaseValue': 0.07523532956838608,
    }

    credibleSetId: Annotated[str, Reference('credible_set', 'id'), PrimaryKey()] = Field(
        pattern=MD5_HEX, description='Identifier of the credible set'
    )
    geneId: Annotated[str, Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Identifier of the gene/target'
    )
    score: float = Field(
        ge=0.0, le=1.0, description=('Locus-to-Gene (L2G) score prediction for the given credibleSetId-geneId pair')
    )
    features: list[Features] = Field(description='List of Locus-to-Gene (L2G) feature values')
    shapBaseValue: Annotated[float, PolarsDtype(pl.Float32)] = Field(ge=0.0, le=1.0, description='SHAP base value')
