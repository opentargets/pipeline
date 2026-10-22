"""ClinGen evidence: the `evidence_clingen` output.

Generated from the 26.09 release output (4,040 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class Urls(StructModel):
    """`urls[]`."""

    niceName: str | None = Field(description='Human readable label of the remote reference')
    url: str = Field(min_length=1, description='URL to remote resource')


class EvidenceClingen(DatasetModel):
    """ClinGen evidence."""

    dataset_name: ClassVar[str] = 'evidence_clingen'
    dataset_title: ClassVar[str] = 'ClinGen evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence from Gene Disease Validity Curation of ClinGen'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'targetFromSourceId': 'ARFGEF1',
        'diseaseFromSourceMappedId': 'MONDO_0859263',
        'datasourceId': 'clingen',
        'datatypeId': 'genetic_literature',
        'diseaseFromSource': (
            'developmental delay, impaired speech, and behavioral abnormalities, with or without seizures'
        ),
        'diseaseFromSourceId': 'MONDO:0859263',
        'allelicRequirements': ['AD'],
        'urls': [
            {
                'niceName': None,
                'url': (
                    'https://search.clinicalgenome.org/kb/gene-validity/CGGV:assertion_03ac7cfe-b728-4549-b2c4-d11b3cfad824-2025-11-21T170000.000Z'
                ),
            }
        ],
        'confidence': 'Definitive',
        'releaseDate': '2025-11-21',
        'studyId': 'Syndromic Disorders Gene Curation Expert Panel',
        'qualityControls': [],
        'diseaseId': 'MONDO_0859263',
        'targetId': 'ENSG00000066777',
        'id': '04e55ac551f7e18f3b4f36bc80f42f724737a5ee',
        'evidenceDate': '2025-11-21',
        'score': 1.0,
    }

    targetFromSourceId: str = Field(
        min_length=1,
        description=(
            'Target ID in resource of origin (accepted sources include Ensembl '
            'gene ID, Uniprot ID, gene symbol), only capital letters are accepted'
        ),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Literal['clingen'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['genetic_literature'] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str = Field(min_length=1, description='Disease identifier from the original source')
    allelicRequirements: list[Literal['AD', 'AR', 'MT', 'SD', 'UD', 'XL']] = Field(
        min_length=1, description='Inheritance patterns'
    )
    urls: list[Urls] = Field(
        min_length=1,
        description=(
            'Reference to linked external resource (e.g. clinical trials, studies, package inserts, reports, etc.)'
        ),
    )
    confidence: Literal[
        'Definitive', 'Disputed', 'Limited', 'Moderate', 'No Known Disease Relationship', 'Refuted', 'Strong'
    ] = Field(description='Confidence qualifier on the reported evidence')
    releaseDate: str = Field(pattern=ISO_DATE, description="Date of the release of the data in a 'YYYY-MM-DD' format")
    studyId: str = Field(min_length=1, description='Identifier of the study generating the data')
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    evidenceDate: str = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.01,
        le=1.0,
        description=('Score of the evidence reflecting the strength of the disease/target relationship'),
    )
