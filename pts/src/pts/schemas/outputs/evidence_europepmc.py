"""EuropePMC evidence: the `evidence_europepmc` output.

Generated from the 26.09 release output (26,583,719 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class TextMiningSentences(StructModel):
    """`textMiningSentences[]`."""

    dEnd: int = Field(ge=2, le=598, description='End position of the disease label in sentence')
    dStart: int = Field(ge=0, le=596, description='Start position of the disease label in the sentence')
    section: Literal[
        'abstract', 'appendix', 'case', 'concl', 'discuss', 'intro', 'other', 'results', 'table', 'title'
    ] = Field(description='Section of the publication the sentence was extracted from')
    tEnd: int = Field(ge=1, le=598, description='End position of the target label in the sentence')
    tStart: int = Field(ge=0, le=596, description='Start position of the target label in the sentence')
    text: str = Field(min_length=1, description='Text of the sentence with the disease/target relationship')


class EvidenceEuropepmc(DatasetModel):
    """EuropePMC evidence."""

    dataset_name: ClassVar[str] = 'evidence_europepmc'
    dataset_title: ClassVar[str] = 'EuropePMC evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence derived from literature mining provided by Europe PMC'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '000092b275b75d97ab2fb377374bdb9acae99dae',
        'targetFromSourceId': 'ENSG00000213921',
        'diseaseFromSourceMappedId': 'MONDO_0017361',
        'datasourceId': 'europepmc',
        'datatypeId': 'literature',
        'resourceScore': 4.0,
        'literature': ['38036973'],
        'textMiningSentences': [
            {
                'dEnd': 36,
                'dStart': 33,
                'section': 'abstract',
                'tEnd': 86,
                'tStart': 81,
                'text': (
                    'Here, we describe a rare case of CRS that occurred in the left kidney '
                    'with a CIC-LEUTX rearrangement.'
                ),
            }
        ],
        'pmcIds': ['PMC10691010'],
        'publicationYear': 2023,
        'qualityControls': [],
        'diseaseId': 'MONDO_0017361',
        'targetId': 'ENSG00000213921',
        'publicationDate': '2023-11-30',
        'evidenceDate': '2023-11-30',
        'score': 0.04,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Literal['europepmc'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['literature'] = Field(description='Type of the evidence')
    resourceScore: float = Field(
        ge=2.0,
        le=2967.0,
        description=('Score provided by datasource indicating strength of target-disease association'),
    )
    literature: list[Annotated[str, Field(min_length=1)]] = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    textMiningSentences: list[TextMiningSentences] = Field(
        min_length=1, description='Sentences of a publication supporting the disease/target relationship'
    )
    pmcIds: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description='List of PubMed Central identifiers of full text publication'
    )
    publicationYear: int = Field(ge=1906, le=2124, description='Year of publication')
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
        ge=0.02,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
