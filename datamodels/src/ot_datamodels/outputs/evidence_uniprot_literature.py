"""Uniprot literature evidence: the `evidence_uniprot_literature` output.

Generated from the 26.12 release output (6,396 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PrimaryKey, Reference
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, enum_of


class EvidenceUniprotLiterature(DatasetModel):
    """Uniprot literature evidence."""

    dataset_name: ClassVar[str] = 'evidence_uniprot_literature'
    dataset_title: ClassVar[str] = 'Uniprot literature evidence'
    dataset_description: ClassVar[str] = 'Protein-disease evidence based on literature curated by the Uniprot team'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '0cc11fe0d231fca7d45c03a0c72243b33d54e8c0',
        'targetFromSourceId': 'O00115',
        'diseaseFromSourceMappedId': 'MONDO_0800132',
        'confidence': 'high',
        'datasourceId': 'uniprot_literature',
        'datatypeId': 'genetic_literature',
        'diseaseFromSource': 'Autoinflammatory-pancytopenia syndrome',
        'diseaseFromSourceId': 'OMIM:619858',
        'literature': ['29259162'],
        'targetModulation': 'up_or_down',
        'qualityControls': [],
        'diseaseId': 'MONDO_0800132',
        'targetId': 'ENSG00000105612',
        'publicationDate': '2017-12-19',
        'evidenceDate': '2017-12-19',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(min_length=1, description='UniProt accession of the target')
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    confidence: Literal['high', 'medium'] = Field(
        description=("Confidence of the UniProt curated gene-disease association ('high', 'medium')")
    )
    datasourceId: Annotated[Literal['uniprot_literature'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    datatypeId: Annotated[Literal['genetic_literature'], enum_of(Datatype)] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease name as curated by UniProt')
    diseaseFromSourceId: str = Field(min_length=1, description='OMIM identifier of the disease as curated by UniProt')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] = Field(
        description='PubMed identifiers cited by UniProt for the gene-disease association'
    )
    targetModulation: str = Field(
        min_length=1, description=("Effect of the disease on the target; 'up_or_down' as UniProt does not specify it")
    )
    qualityControls: list[str] = Field(max_length=0, description='Evidence quality flags')
    diseaseId: Annotated[str, Reference('disease', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ONTOLOGY_ID, description='Open Targets disease identifier'
    )
    targetId: Annotated[str, Reference('target', 'id'), ClickHouseType('LowCardinality(String)')] = Field(
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
            'relationship, derived from the UniProt confidence'
        ),
    )
