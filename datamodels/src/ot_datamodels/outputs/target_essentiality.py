"""Target - DepMap essentiality: the `target_essentiality` output.

Generated from the 26.12 release output (18,453 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, ONTOLOGY_ID


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
        description='Gene effect score indicating the impact of gene knockout'
    )
    expression: Annotated[float, PolarsDtype(pl.Float32)] | None = Field(
        ge=0.0, description='Gene expression level in the corresponding cell line'
    )


class DepMapEssentiality(StructModel):
    """`depMapEssentiality[]`."""

    tissueId: Annotated[str, Bioregistry('uberon'), Reference('biosample', 'biosampleId')] | None = Field(
        pattern=ONTOLOGY_ID, description='Identifier of the tissue from which the cells were sampled for assay'
    )
    tissueName: str = Field(min_length=1, description='Name of the tissue from which the cells were sample for assay')
    screens: list[Screens] = Field(
        description=('List of CRISPR screening experiments supporting the essentiality assessment')
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
        'targetId': 'ENSG00000000003',
        'isEssential': False,
        'depMapEssentiality': [
            {
                'tissueId': 'CL_0000057',
                'tissueName': 'other',
                'screens': [
                    {
                        'depmapId': 'ACH-000025',
                        'cellLineName': 'CH-157MN',
                        'diseaseFromSource': 'Meningothelial Tumor',
                        'diseaseCellLineId': 'SIDM01193',
                        'mutation': 'damaging',
                        'geneEffect': 0.0018769027665257454,
                        'expression': 4.4099884033203125,
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
        description=('Essentiality measurements extraced from DepMap, stratified by tissue or anatomical units')
    )
