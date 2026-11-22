"""Orphanet evidence: the `evidence_orphanet` output.

Generated from the 26.12 release output (6,781 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import DatasetModel, PolarsDtype, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, DirectionOnTarget, DirectionOnTrait, enum_of


class EvidenceOrphanet(DatasetModel):
    """Orphanet evidence."""

    dataset_name: ClassVar[str] = 'evidence_orphanet'
    dataset_title: ClassVar[str] = 'Orphanet evidence'
    dataset_description: ClassVar[str] = 'Curated collection of rare disase-target evidence from Orphanet'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000166813',
        'id': '89ffd329edf82620bc70a14a679cafc8c05105ff',
        'targetFromSourceId': 'ENSG00000166813',
        'diseaseFromSourceMappedId': 'MONDO_0011778',
        'datasourceId': 'orphanet',
        'datatypeId': 'genetic_association',
        'alleleOrigins': ['germline'],
        'confidence': 'Assessed',
        'diseaseFromSource': 'Multiple epiphyseal dysplasia-macrocephaly-facial dysmorphism syndrome',
        'diseaseFromSourceId': 'Orphanet_166024',
        'literature': ['22587682'],
        'targetFromSource': 'kinesin family member 7',
        'variantFunctionalConsequenceId': 'SO_0002054',
        'qualityControls': [],
        'diseaseId': 'MONDO_0011778',
        'publicationDate': '2012-05-15',
        'evidenceDate': '2012-05-15',
        'score': 1.0,
        'directionOnTrait': 'risk',
        'directionOnTarget': 'LoF',
    }

    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target as reported by Orphanet'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Annotated[Literal['orphanet'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    datatypeId: Annotated[Datatype, PolarsDtype(pl.Enum)] = Field(description='Type of the evidence')
    alleleOrigins: list[Literal['germline', 'somatic']] = Field(
        description=('Origin of the variant allele; germline and somatic for all Orphanet associations')
    )
    confidence: Literal['Assessed', 'Not yet assessed'] = Field(
        description=("Orphanet assessment status of the gene-disease association ('Assessed', 'Not yet assessed')")
    )
    diseaseFromSource: str = Field(min_length=1, description='Name of the Orphanet disorder')
    diseaseFromSourceId: str = Field(pattern=ONTOLOGY_ID, description='Orphanet identifier of the disorder')
    literature: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='PubMed identifiers supporting the Orphanet gene-disease association'
    )
    targetFromSource: str = Field(min_length=1, description='Gene name as reported by Orphanet')
    variantFunctionalConsequenceId: Annotated[str, Reference('so', 'id')] | None = Field(
        pattern=ONTOLOGY_ID,
        description=(
            'Sequence ontology (SO) identifier of the variant effect (loss or gain '
            'of function) derived from the Orphanet association type'
        ),
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship, derived from the Orphanet assessment status'
        ),
    )
    directionOnTrait: Annotated[DirectionOnTrait, PolarsDtype(pl.Enum)] = Field(
        description='Predicted direction of effect on the trait'
    )
    directionOnTarget: Annotated[DirectionOnTarget, PolarsDtype(pl.Enum)] | None = Field(
        description=(
            'Gain or loss of function effect of the evidence on the target '
            'resulting from genetic variants, pharmacological modulation, or other '
            'perturbations'
        )
    )
