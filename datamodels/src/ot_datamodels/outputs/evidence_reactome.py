"""Reactome evidence: the `evidence_reactome` output.

Generated from the 26.12 release output (10,749 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseType, DatasetModel, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX
from ot_datamodels.vocabularies import Datasource, Datatype, enum_of


class Pathways(StructModel):
    """`pathways[]`."""

    id: Annotated[str, Bioregistry('reactome')] = Field(min_length=1, description='Reactome pathway identifier')
    name: str = Field(min_length=1, description='Name of the Reactome pathway')


class EvidenceReactome(DatasetModel):
    """Reactome evidence."""

    dataset_name: ClassVar[str] = 'evidence_reactome'
    dataset_title: ClassVar[str] = 'Reactome evidence'
    dataset_description: ClassVar[str] = (
        'Manually curated disease-target evidence based on reaction pathways sourced from Reactome'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '4c5190e8d33cc6dc0e8e6b2a5b6d0bd7e12177b9',
        'targetFromSourceId': 'P00533',
        'diseaseFromSourceMappedId': 'EFO_0000311',
        'datasourceId': 'reactome',
        'datatypeId': 'affected_pathway',
        'diseaseFromSource': 'cancer',
        'diseaseFromSourceId': 'DOID:162',
        'literature': ['17177598'],
        'pathways': [
            {'id': 'R-HSA-1236382', 'name': 'Constitutive Signaling by Ligand-Responsive EGFR Cancer Variants'}
        ],
        'reactionId': 'R-HSA-1169421',
        'reactionName': ('Trans-autophosphorylation of activated ligand-responsive EGFR mutant dimers'),
        'targetModulation': 'gain_of_function',
        'variantAminoacidDescriptions': ['Deletion of residues 747 to 752'],
        'qualityControls': [],
        'diseaseId': 'MONDO_0004992',
        'targetId': 'ENSG00000146648',
        'publicationDate': '2004-07-29',
        'evidenceDate': '2004-07-29',
        'score': 1.0,
    }

    id: Annotated[str, PrimaryKey()] = Field(pattern=SHA1_HEX, description='Identifer of the disease/target evidence')
    targetFromSourceId: str = Field(
        min_length=1, description='UniProt accession of the target as annotated in Reactome'
    )
    diseaseFromSourceMappedId: str = Field(pattern=ONTOLOGY_ID, description='Mapped Open Targets disease identifier')
    datasourceId: Annotated[Literal['reactome'], enum_of(Datasource)] = Field(
        description='Identifer of the evidence source'
    )
    datatypeId: Annotated[Literal['affected_pathway'], enum_of(Datatype)] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label as annotated in Reactome')
    diseaseFromSourceId: str = Field(
        min_length=1, description='Disease Ontology (DOID) identifier as annotated in Reactome'
    )
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        description='PubMed identifiers supporting the Reactome reaction'
    )
    pathways: list[Pathways] = Field(description='Reactome pathways associated with the reaction')
    reactionId: str = Field(min_length=1, description='Reactome identifier of the disease reaction')
    reactionName: str = Field(min_length=1, description='Name of the Reactome disease reaction')
    targetModulation: str = Field(
        min_length=1,
        description=(
            'Effect of the disease variant on the target as annotated in Reactome '
            "(e.g. 'loss_of_function', 'gain_of_function')"
        ),
    )
    variantAminoacidDescriptions: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description=('Protein-level descriptions of the disease variants annotated in Reactome')
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
            'relationship; 1.0 for all Reactome evidence'
        ),
    )
