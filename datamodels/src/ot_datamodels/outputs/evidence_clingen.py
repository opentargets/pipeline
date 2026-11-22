"""ClinGen evidence: the `evidence_clingen` output.

Generated from the 26.12 release output (4,103 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, enum_of


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
        'targetFromSourceId': 'AARS1',
        'diseaseFromSourceMappedId': 'MONDO_0013212',
        'datasourceId': 'clingen',
        'datatypeId': 'genetic_literature',
        'diseaseFromSource': 'Charcot-Marie-Tooth disease axonal type 2N',
        'diseaseFromSourceId': 'MONDO:0013212',
        'allelicRequirements': ['AD'],
        'urls': [
            {
                'niceName': None,
                'url': (
                    'https://search.clinicalgenome.org/kb/gene-validity/CGGV:assertion_92de3832-c272-4993-8586-288c6331dec2-2024-03-14T160000.000Z'
                ),
            }
        ],
        'confidence': 'Definitive',
        'releaseDate': '2024-03-14',
        'studyId': 'Charcot-Marie-Tooth Disease Gene Curation Expert Panel',
        'qualityControls': [],
        'diseaseId': 'MONDO_0013212',
        'targetId': 'ENSG00000090861',
        'id': '36a4013139084580ce238a84e551fd56bb91faa1',
        'evidenceDate': '2024-03-14',
        'score': 1.0,
    }

    targetFromSourceId: str = Field(min_length=1, description='Gene symbol of the target as curated by ClinGen')
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Annotated[Literal['clingen'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    datatypeId: Annotated[Literal['genetic_literature'], enum_of(Datatype)] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label as curated by ClinGen')
    diseaseFromSourceId: str = Field(min_length=1, description='MONDO identifier of the disease as curated by ClinGen')
    allelicRequirements: list[Annotated[str, Field(min_length=1)]] = Field(
        description=('Mode of inheritance of the gene-disease relationship as curated by ClinGen (e.g. AD, AR, XL)')
    )
    urls: list[Urls] = Field(description='Link to the ClinGen gene-disease validity curation report')
    confidence: str = Field(
        min_length=1,
        description=(
            "ClinGen gene-disease validity classification (e.g. 'Definitive', 'Moderate', 'Limited', 'Refuted')"
        ),
    )
    releaseDate: str = Field(
        pattern=ISO_DATE,
        description=("Date of the ClinGen gene-disease validity classification in a 'YYYY-MM-DD' format"),
    )
    studyId: str = Field(
        min_length=1, description=('Name of the ClinGen Gene Curation Expert Panel (GCEP) that made the classification')
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Open Targets target identifier'
    )
    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    evidenceDate: str = Field(pattern=ISO_DATE, description='Earliest data for the evidence')
    score: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship, derived from the ClinGen classification'
        ),
    )
