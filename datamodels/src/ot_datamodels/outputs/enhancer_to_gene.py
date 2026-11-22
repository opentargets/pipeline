"""Enhancer to gene: the `enhancer_to_gene` output.

Generated from the 26.12 release output (48,799,585 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseType, DatasetModel, PolarsDtype, Reference, StructModel
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, MD5_HEX, ONTOLOGY_ID
from ot_datamodels.vocabularies import Chromosome, IntervalDatasource, IntervalType


class ResourceScore(StructModel):
    """`resourceScore[]`."""

    name: Annotated[Literal['DNase', 'HiC_contacts'], ClickHouseType('LowCardinality(String)')] = Field(
        description=('Individual components which contributes to the interval score (e.g., DNase, HiC_contacts)')
    )
    value: Annotated[float, PolarsDtype(pl.Float32), ClickHouseType('Float64')] = Field(
        ge=0.0, le=1.0, description='Raw signal value from the resourceScore(s)'
    )


class EnhancerToGene(DatasetModel):
    """Enhancer to gene."""

    dataset_name: ClassVar[str] = 'enhancer_to_gene'
    dataset_title: ClassVar[str] = 'Enhancer to gene'
    dataset_description: ClassVar[str] = 'Genomic regions associated with genes based on epigenetics evidence'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target', 'Genetics')
    example: ClassVar[dict] = {
        'chromosome': '1',
        'start': 778366,
        'end': 778866,
        'geneId': 'ENSG00000228794',
        'score': 0.645621568911758,
        'distanceToTss': 48436,
        'resourceScore': [{'name': 'DNase', 'value': 0.30818700790405273}],
        'datasourceId': 'E2G',
        'intervalType': 'intergenic',
        'pmid': '38014075',
        'biosampleName': 'right kidney',
        'biosampleFromSourceId': None,
        'biosampleId': 'UBERON_0002113',
        'studyId': 'ENCFF229GDZ',
        'intervalId': '349400b9fa555cb9c170e98971eb7173',
        # a valid interval may carry a flag the step does not exclude by; one is here so the element rules run
        'qualityControls': ['Interval has a duplicate with different score'],
    }

    chromosome: Annotated[Chromosome, PolarsDtype(pl.Enum)] = Field(description='Chromosome the interval is located on')
    start: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='Basepair position of the interval start'
    )
    end: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='Basepair position of the interval end'
    )
    geneId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target the interval is linked with'
    )
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Interval evidence score from the datasource, higher values indicate '
            'stronger support linking the interval to its target gene'
        ),
    )
    distanceToTss: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=0, description='Distance of the interval from the transcription start site of the gene'
    )
    resourceScore: list[ResourceScore] = Field(description='List of contributing score components')
    datasourceId: Annotated[IntervalDatasource, PolarsDtype(pl.Enum)] = Field(
        description='Identifier of the interval datasource'
    )
    intervalType: Annotated[IntervalType, PolarsDtype(pl.Enum)] = Field(description='Type of the interval')
    pmid: str = Field(
        pattern=DIGITS, description='Pubmed identifier of the publication that reported the interval data'
    )
    biosampleName: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description='Name of the cell type/anatomical unit the interval was identified in'
    )
    biosampleFromSourceId: str | None = Field(
        description=('Original identifier of the cell type/anatomical unit the interval was identified in')
    )
    biosampleId: Annotated[str, Reference('biosample', 'biosampleId'), ClickHouseType('LowCardinality(String)')] = (
        Field(pattern=ONTOLOGY_ID, description='Identifier of the biosample in where the interval was discovered in')
    )
    studyId: str = Field(min_length=1, description='Identifier of the interval study')
    intervalId: str = Field(pattern=MD5_HEX, description='Identifier of the interval')
    qualityControls: list[
        Literal[
            'Target/gene identifier could not match to reference',
            'Biosample identifier was not found in the reference',
            'Score was above or below specified thresholds',
            'Interval type is not supported',
            'Interval has a duplicate with different score',
            'Interval chromosome was not found in contig index',
            'Interval range exceeded chromosome bounds',
            'Multiple interval types for the same (region, geneId) pair',
            'Interval chromosome does not match associated gene chromosome',
        ]
    ] = Field(description='Quality control flags for this interval')
