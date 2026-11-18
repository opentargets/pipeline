"""EuropePMC evidence: the `evidence_europepmc` output.

Generated from the 26.12 release output (28,772,039 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class TextMiningSentences(StructModel):
    """`textMiningSentences[]`."""

    dEnd: int = Field(ge=0, description='End position of the disease label in sentence')
    dStart: int = Field(ge=0, description='Start position of the disease label in the sentence')
    section: Literal[
        'abstract', 'appendix', 'case', 'concl', 'discuss', 'figure', 'intro', 'other', 'results', 'table', 'title'
    ] = Field(description='Section of the publication the sentence was extracted from')
    tEnd: int = Field(ge=0, description='End position of the target label in the sentence')
    tStart: int = Field(ge=0, description='Start position of the target label in the sentence')
    text: str = Field(min_length=1, description='Text of the sentence with the disease/target relationship')


class EvidenceEuropepmc(DatasetModel):
    """EuropePMC evidence."""

    dataset_name: ClassVar[str] = 'evidence_europepmc'
    dataset_title: ClassVar[str] = 'EuropePMC evidence'
    dataset_description: ClassVar[str] = 'Disease-target evidence derived from literature mining provided by Europe PMC'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '3a7133c95f13a257c2f735480dcf81ecdfba2f4c',
        'targetFromSourceId': 'ENSG00000160791',
        'diseaseFromSourceMappedId': 'EFO_0000180',
        'datasourceId': 'europepmc',
        'datatypeId': 'literature',
        'resourceScore': 3.0,
        'literature': ['15577536'],
        'textMiningSentences': [
            {
                'dEnd': 423,
                'dStart': 408,
                'section': 'abstract',
                'tEnd': 222,
                'tStart': 218,
                'text': (
                    'This latter pair are poorly expressed at the cell surface, weakly '
                    'bind macrophage inflammatory protein 1beta (CCL4) and RANTES (CCL5), '
                    'and display reduced HIV-1 coreceptor '
                    'efficiency.<h4>Conclusions</h4>Among the five CCR5 variants found in '
                    'this Vietnamese population, G106R and C269F displayed significant '
                    'modifications of their receptor and coreceptor properties, which may '
                    'contribute to susceptibility to HIV-1 infection and/or disease '
                    'progression within this population.'
                ),
            }
        ],
        'pmcIds': ['PMC4059663'],
        'publicationYear': 2004,
        'qualityControls': [],
        'diseaseId': 'EFO_0000180',
        'targetId': 'ENSG00000160791',
        'publicationDate': '2004-11-01',
        'evidenceDate': '2004-11-01',
        'score': 0.03,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        pattern=ENSEMBL_GENE_ID,
        description=('Ensembl gene identifier of the target recognised by Europe PMC text mining'),
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Literal['europepmc'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['literature'] = Field(description='Type of the evidence')
    resourceScore: float = Field(
        ge=0.0,
        description=('Sum of the Europe PMC text-mining scores of the target-disease co-mentions in the publication'),
    )
    literature: list[Annotated[str, Field(min_length=1)]] = Field(
        description=(
            'PubMed or preprint identifier of the publication in which the target and the disease are co-mentioned'
        )
    )
    textMiningSentences: list[TextMiningSentences] = Field(
        description=('Sentences of the publication in which the target and the disease are co-mentioned')
    )
    pmcIds: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description='PubMed Central identifiers of the full text publication'
    )
    publicationYear: int | None = Field(ge=0, description='Year of publication of the article')
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
        ge=0.0,
        le=1.0,
        description=(
            'Score of the evidence reflecting the strength of the disease/target '
            'relationship; the Europe PMC resource score divided by 100, capped at '
            '1'
        ),
    )
