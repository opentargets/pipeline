"""Enhancer to gene: the `enhancer_to_gene` output.

Generated from the 26.09 release output (48,808,491 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, MD5_HEX, ONTOLOGY_ID


class ResourceScore(StructModel):
    """`resourceScore[]`."""

    name: Literal['DNase', 'HiC_contacts'] = Field(
        description=('Individual components which contributes to the interval score (e.g., DNase, HiC_contacts)')
    )
    value: Annotated[float, PolarsDtype(pl.Float32)] = Field(
        ge=0.0003549999964889139, le=1.0, description='Raw signal value from the resourceScore(s)'
    )


class EnhancerToGene(DatasetModel):
    """Enhancer to gene."""

    dataset_name: ClassVar[str] = 'enhancer_to_gene'
    dataset_title: ClassVar[str] = 'Enhancer to gene'
    dataset_description: ClassVar[str] = 'Genomic regions associated with genes based on epigenetics evidence'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target', 'Genetics')
    example: ClassVar[dict] = {
        'biosampleId': 'UBERON_0002113',
        'geneId': 'ENSG00000225880',
        'chromosome': '1',
        'start': 778366,
        'end': 778866,
        'score': 0.645833630694996,
        'distanceToTss': 48930,
        'resourceScore': [{'name': 'DNase', 'value': 0.30818700790405273}],
        'datasourceId': 'E2G',
        'intervalType': 'intergenic',
        'pmid': '38014075',
        'biosampleName': 'right kidney',
        'biosampleFromSourceId': None,
        'studyId': 'ENCFF229GDZ',
        'intervalId': 'c89213ff6f21bcc0326986caac60b8d7',
        'qualityControls': [],
    }

    biosampleId: Annotated[str, Reference('biosample', 'biosampleId')] = Field(
        pattern=ONTOLOGY_ID, description='Identifier of the biosample in where the interval was discovered in'
    )
    geneId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target the interval is linked with'
    )
    chromosome: str = Field(min_length=1, description='Chromosome the interval is located on')
    start: int = Field(ge=10107, le=248924588, description='Basepair position of the interval start')
    end: int = Field(ge=10859, le=248925639, description='Basepair position of the interval end')
    score: float = Field(
        ge=0.600000002954682,
        le=0.9999999988532128,
        description=(
            'Interval evidence score from the datasource, higher values indicate '
            'stronger support linking the interval to its target gene'
        ),
    )
    distanceToTss: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=0, le=55760344, description='Distance of the interval from the transcription start site of the gene'
    )
    resourceScore: list[ResourceScore] = Field(min_length=1, description='List of contributing score components')
    datasourceId: Literal['E2G'] = Field(description='Identifier of the interval datasource')
    intervalType: Literal['genic', 'intergenic', 'promoter'] = Field(description='Type of the interval')
    pmid: Literal['38014075'] = Field(
        description='Pubmed identifier of the publication that reported the interval data'
    )
    biosampleName: str = Field(
        min_length=1, description='Name of the cell type/anatomical unit the interval was identified in'
    )
    biosampleFromSourceId: str | None = Field(
        description=('Original identifier of the cell type/anatomical unit the interval was identified in')
    )
    studyId: str = Field(min_length=1, description='Identifier of the interval study')
    intervalId: str = Field(pattern=MD5_HEX, description='Identifier of the interval')
    qualityControls: list[str] = Field(max_length=0, description='Quality control flags for this interval')
