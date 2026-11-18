"""Genomics England - PanelApp evidence: the `evidence_genomics_england` output.

Generated from the 26.12 release output (42,560 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class EvidenceGenomicsEngland(DatasetModel):
    """Genomics England - PanelApp evidence."""

    dataset_name: ClassVar[str] = 'evidence_genomics_england'
    dataset_title: ClassVar[str] = 'Genomics England - PanelApp evidence'
    dataset_description: ClassVar[str] = 'Curated disease-target evidence provided by Genomics England PanelApp'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '7e68e07a145b2aaa83d7a82d59b819f8b001833d',
        'targetFromSourceId': 'CBS',
        'diseaseFromSourceMappedId': 'MONDO_0007947',
        'studyId': '1',
        'studyOverview': 'Thoracic aortic aneurysm or dissection',
        'confidence': 'amber',
        'cohortPhenotypes': ['Marfan syndrome'],
        'diseaseFromSource': 'Marfan syndrome',
        'diseaseFromSourceId': 'OMIM:617602',
        'literature': ['28288113'],
        'allelicRequirements': ['BIALLELIC, autosomal or pseudoautosomal'],
        'datasourceId': 'genomics_england',
        'datatypeId': 'genetic_literature',
        'qualityControls': [],
        'diseaseId': 'MONDO_0007947',
        'targetId': 'ENSG00000160200',
        'publicationDate': '2017-03-13',
        'evidenceDate': '2017-03-13',
        'score': 0.5,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1, description='Gene symbol of the target as reported in Genomics England PanelApp'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    studyId: str = Field(pattern=DIGITS, description='Genomics England PanelApp panel identifier')
    studyOverview: str = Field(min_length=1, description='Name of the Genomics England PanelApp panel')
    confidence: Literal['amber', 'green'] = Field(
        description=("Genomics England PanelApp gene rating ('green': high evidence, 'amber': moderate evidence)")
    )
    cohortPhenotypes: list[str] = Field(
        description=('Phenotypes associated with the gene in the Genomics England PanelApp panel')
    )
    diseaseFromSource: str | None = Field(
        min_length=1, description='Phenotype label as curated in the Genomics England PanelApp panel'
    )
    diseaseFromSourceId: str | None = Field(
        min_length=1,
        description=('OMIM, Orphanet, HPO or MONDO identifier extracted from the Genomics England PanelApp phenotype'),
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description=('PubMed identifiers listed for the gene in the Genomics England PanelApp panel')
    )
    allelicRequirements: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description=('Mode of inheritance of the gene as curated in the Genomics England PanelApp panel')
    )
    datasourceId: str = Field(pattern=ONTOLOGY_ID, description='Identifer of the evidence source')
    datatypeId: str = Field(pattern=ONTOLOGY_ID, description='Type of the evidence')
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
            'relationship, derived from the PanelApp gene rating'
        ),
    )
