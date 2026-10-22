"""Uniprot literature evidence: the `evidence_uniprot_literature` output.

Generated from the 26.09 release output (6,394 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceUniprotLiterature(DatasetModel):
    """Uniprot literature evidence."""

    dataset_name: ClassVar[str] = 'evidence_uniprot_literature'
    dataset_title: ClassVar[str] = 'Uniprot literature evidence'
    dataset_description: ClassVar[str] = 'Protein-disease evidence based on literature curated by the Uniprot team'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '5081c7eafe29941530d230601ed5bf22fe66f140',
        'targetFromSourceId': 'Q9H257',
        'diseaseFromSourceMappedId': 'Orphanet_1334',
        'confidence': 'high',
        'datasourceId': 'uniprot_literature',
        'datatypeId': 'genetic_literature',
        'diseaseFromSource': 'Immunodeficiency 103, susceptibility to fungal infections',
        'diseaseFromSourceId': 'OMIM:212050',
        'literature': ['26941346'],
        'targetModulation': 'up_or_down',
        'qualityControls': [],
        'diseaseId': 'MONDO_0015279',
        'targetId': 'ENSG00000187796',
        'publicationDate': '2009-10-01',
        'evidenceDate': '2009-10-01',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    confidence: Literal['high', 'medium'] = Field(description='Confidence qualifier on the reported evidence')
    datasourceId: Literal['uniprot_literature'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_literature'] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str = Field(min_length=1, description='Disease identifier from the original source')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] = Field(
        description='List of PubMed or preprint reference identifiers'
    )
    targetModulation: Literal['up_or_down'] = Field(description='Description of target modulation event')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    publicationDate: str | None = Field(
        pattern=ISO_DATE, description='Date of the earliest publication supporting the evidence'
    )
    evidenceDate: str | None = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.5, le=1.0, description=('Score of the evidence reflecting the strength of the disease/target relationship')
    )
