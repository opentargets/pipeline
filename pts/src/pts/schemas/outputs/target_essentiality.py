"""Target - DepMap essentiality: the `target_essentiality` output.

Generated from the 26.09 release output (18,460 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


class Screens(StructModel):
    """`depMapEssentiality[]/screens[]`."""

    depmapId: str = Field(min_length=1, description='Unique identifier of the assay in DepMap')
    cellLineName: str = Field(
        min_length=1, description=('Name of the cancer cell line in which the gene essentiality was assessed')
    )
    diseaseFromSource: str = Field(
        min_length=1, description='Disease associated with the cell line as reported in the source data'
    )
    diseaseCellLineId: str | None = Field(
        min_length=1, description='Cell model passport identifier of a cell line modelling a disease'
    )
    mutation: Literal['damaging', 'hotspot'] | None = Field(description='Background mutation in the tested cell line')
    geneEffect: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=-6.258358478546143,
        le=5.769171714782715,
        description='Gene effect score indicating the impact of gene knockout',
    )
    expression: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, le=15.37379264831543, description='Gene expression level in the corresponding cell line'
    )


class DepMapEssentiality(StructModel):
    """`depMapEssentiality[]`."""

    tissueId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] | None = Field(
        pattern=ONTOLOGY_ID, description='Identifier of the tissue from which the cells were sampled for assay'
    )
    tissueName: str = Field(min_length=1, description='Name of the tissue from which the cells were sample for assay')
    screens: list[Screens] = Field(
        min_length=1, description=('List of CRISPR screening experiments supporting the essentiality assessment')
    )


class TargetEssentiality(DatasetModel):
    """Target - DepMap essentiality."""

    dataset_name: ClassVar[str] = 'target_essentiality'
    dataset_title: ClassVar[str] = 'Target - DepMap essentiality'
    dataset_description: ClassVar[str] = (
        'Target essentiality assessments based on dependencies exhibited when '
        'knocking-out genes in cancer cellular models using CRISPR screenings '
        'as reported in by the Cancer DepMap Project. Gene effects below -1 '
        'can be considered dependencies'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000001561',
        'isEssential': False,
        'depMapEssentiality': [
            {
                'tissueId': 'UBERON_0001264',
                'tissueName': 'pancreas',
                'screens': [
                    {
                        'depmapId': 'ACH-000517',
                        'cellLineName': 'SNU-410',
                        'diseaseFromSource': 'Pancreatic Adenocarcinoma',
                        'diseaseCellLineId': 'SIDM00505',
                        'mutation': 'damaging',
                        'geneEffect': 0.18275734782218933,
                        'expression': 1.5346941947937012,
                    }
                ],
            }
        ],
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id'), PrimaryKey()] = Field(
        pattern=ENSEMBL_GENE_ID, description='Target identifier the essentiality was measured on'
    )
    isEssential: bool = Field(
        description=('Boolean flag indicating whether the tested gene is considered essential in the given context')
    )
    depMapEssentiality: list[DepMapEssentiality] = Field(
        min_length=1,
        description=('Essentiality measurements extraced from DepMap, stratified by tissue or anatomical units'),
    )
