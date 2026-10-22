"""Reactome evidence: the `evidence_reactome` output.

Generated from the 26.09 release output (10,749 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import DIGITS, ENSEMBL_GENE_ID, ISO_DATE, ONTOLOGY_ID, SHA1_HEX


class Pathways(StructModel):
    """`pathways[]`."""

    id: Annotated[str, Bioregistry('reactome')] = Field(min_length=1, description='Reactome pathway identifier')
    name: str = Field(min_length=1, description='Pathway or gene set/pathway name')


class EvidenceReactome(DatasetModel):
    """Reactome evidence."""

    dataset_name: ClassVar[str] = 'evidence_reactome'
    dataset_title: ClassVar[str] = 'Reactome evidence'
    dataset_description: ClassVar[str] = (
        'Manually curated disease-target evidence based on reaction pathways sourced from Reactome'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target-Disease',)
    example: ClassVar[dict] = {
        'id': '3274096a8dcdc95e2f4d35f6138388ec94ce53b3',
        'targetFromSourceId': 'P63165',
        'diseaseFromSourceMappedId': 'MONDO_0100096',
        'datasourceId': 'reactome',
        'datatypeId': 'affected_pathway',
        'diseaseFromSource': 'COVID-19',
        'diseaseFromSourceId': 'DOID:0080600',
        'literature': ['15848177'],
        'pathways': [{'id': 'R-HSA-9694631', 'name': 'Maturation of nucleoprotein'}],
        'reactionId': 'R-HSA-9729307',
        'reactionName': 'Nucleoprotein is SUMOylated',
        'targetModulation': 'up_or_down',
        'variantAminoacidDescriptions': ['glycine 106 replaced with L-arginine'],
        'qualityControls': [],
        'diseaseId': 'MONDO_0100096',
        'targetId': 'ENSG00000116030',
        'publicationDate': '2005-04-01',
        'evidenceDate': '2005-04-01',
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
    datasourceId: Literal['reactome'] = Field(description='Identifer of the evidence source')
    datatypeId: Literal['affected_pathway'] = Field(description='Type of the evidence')
    diseaseFromSource: str = Field(min_length=1, description='Disease label from the original source')
    diseaseFromSourceId: str = Field(min_length=1, description='Disease identifier from the original source')
    literature: list[Annotated[str, Field(pattern=DIGITS)]] | None = Field(
        min_length=1, description='List of PubMed or preprint reference identifiers'
    )
    pathways: list[Pathways] = Field(min_length=1, description='List of pooled pathways')
    reactionId: str = Field(min_length=1, description='Pathway, gene set or reaction identifier in Reactome')
    reactionName: str = Field(min_length=1, description='Name of the reaction, patway or gene set in Reactome')
    targetModulation: Literal[
        'decreased_transcript_level',
        'gain_of_function',
        'increased_translational_product_level',
        'loss_of_function',
        'partial_loss_of_function',
        'up_or_down',
    ] = Field(description='Description of target modulation event')
    variantAminoacidDescriptions: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description='Descriptions of variant consequences at protein level'
    )
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
        ge=1.0, le=1.0, description=('Score of the evidence reflecting the strength of the disease/target relationship')
    )
