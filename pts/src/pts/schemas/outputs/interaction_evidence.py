"""Molecular interactions evidence: the `interaction_evidence` output.

Generated from the 26.12 release output (22,620,749 rows) and its croissant metadata by
`scripts/output_schemas/generate.py`, whose docstring lists which observations become rules.
"""

from typing import Annotated, ClassVar, Literal

from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, Reference, StructModel


class InteractionResources(StructModel):
    """`interactionResources`."""

    databaseVersion: str = Field(
        min_length=1, description='Version of the source database providing interaction evidence'
    )
    sourceDatabase: Literal['intact', 'reactome', 'signor', 'string'] = Field(
        description='Name of the source database reporting the interaction evidence'
    )


class HostOrganismTissue(StructModel):
    """`hostOrganismTissue`."""

    fullName: str | None = Field(min_length=1, description='Full name of the tissue')
    shortName: str | None = Field(min_length=1, description='Short name of the tissue')
    xrefs: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description="Cross-references to external databases for the host organism's tissue"
    )


class ParticipantDetectionMethodA(StructModel):
    """`participantDetectionMethodA[]`."""

    miIdentifier: Annotated[str, Bioregistry('mi')] | None = Field(
        min_length=1, description=("Molecular Interactions (MI) identifier for participant A's detection method")
    )
    shortName: str | None = Field(min_length=1, description="Short name of participant A's detection method")


class ParticipantDetectionMethodB(StructModel):
    """`participantDetectionMethodB[]`."""

    miIdentifier: Annotated[str, Bioregistry('mi')] | None = Field(
        min_length=1, description=("Molecular Interactions (MI) identifier for participant B's detection method")
    )
    shortName: str | None = Field(min_length=1, description="Short name of participant B's detection method")


class InteractionEvidence(DatasetModel):
    """Molecular interactions evidence."""

    dataset_name: ClassVar[str] = 'interaction_evidence'
    dataset_title: ClassVar[str] = 'Molecular interactions evidence'
    dataset_description: ClassVar[str] = (
        "Experimental evidence supporting molecular interactions ('interaction') dataset"
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'interactionResources': {'databaseVersion': '253', 'sourceDatabase': 'intact'},
        'interactionScore': 0.28,
        'interactionDetectionMethodShortName': 'mirna interference luciferase assay',
        'interactionDetectionMethodMiIdentifier': 'MI:2285',
        'evidenceScore': 0.075,
        'interactionIdentifier': 'EBI-16765507',
        'pubmedId': '23426184',
        'interactionId': 1561581066758384050,
        'expansionMethodMiIdentifier': 'MI:1060',
        'expansionMethodShortName': 'spoke expansion',
        'hostOrganismScientificName': 'Homo sapiens HEK293 embryonic kidney cell',
        'hostOrganismTaxId': 9606,
        'hostOrganismTissue': {
            'fullName': 'bronchial epithelium',
            'shortName': 'bronchial_epithelium',
            'xrefs': ['BTO:0004296'],
        },
        'interactionTypeMiIdentifier': 'MI:0915',
        'interactionTypeShortName': 'physical association',
        'participantDetectionMethodA': [{'miIdentifier': 'MI:1220', 'shortName': 'tag luciferase'}],
        'participantDetectionMethodB': [{'miIdentifier': 'MI:1220', 'shortName': 'tag luciferase'}],
    }

    interactionResources: InteractionResources = Field(description='Databases providing evidence for the interaction')
    interactionScore: float | None = Field(
        ge=0.0, le=1.0, description='Score assigned to the interaction based on supporting evidence'
    )
    interactionDetectionMethodShortName: str = Field(
        min_length=1, description='Short name of the method used to detect the interaction'
    )
    interactionDetectionMethodMiIdentifier: Annotated[str, Bioregistry('mi')] = Field(
        description=('Molecular Interactions (MI) identifier for the interaction detection method')
    )
    evidenceScore: float | None = Field(
        ge=0.0, le=1.0, description=('Score indicating the confidence or strength of the interaction evidence')
    )
    interactionIdentifier: str | None = Field(
        min_length=1, description='Unique identifier for the interaction evidence entry at the source'
    )
    pubmedId: str | None = Field(
        min_length=1, description='PubMed ID of the publication supporting the interaction evidence'
    )
    interactionId: Annotated[int, Reference('interaction', 'interactionId')] = Field(
        description='Identifier of the molecular interaction this evidence supports'
    )
    expansionMethodMiIdentifier: Annotated[Literal['MI:1060', 'MI:1061'], Bioregistry('mi')] | None = Field(
        description='Molecular Interactions (MI) identifier for the expansion method used'
    )
    expansionMethodShortName: Literal['matrix expansion', 'spoke expansion'] | None = Field(
        description='Short name of the method used to expand the interaction dataset'
    )
    hostOrganismScientificName: str | None = Field(
        min_length=1, description=('Scientific name of the host organism in which the interaction was observed')
    )
    hostOrganismTaxId: int | None = Field(description='NCBI taxon ID of the host organism')
    hostOrganismTissue: HostOrganismTissue | None = Field(
        description='Tissue of the host organism where the interaction was identified'
    )
    interactionTypeMiIdentifier: Annotated[str, Bioregistry('mi')] | None = Field(
        min_length=1, description='Molecular Interactions (MI) identifier for the type of interaction'
    )
    interactionTypeShortName: str | None = Field(min_length=1, description='Short name of the interaction type')
    participantDetectionMethodA: list[ParticipantDetectionMethodA] | None = Field(
        description='Detection method used to identify participant A in the interaction'
    )
    participantDetectionMethodB: list[ParticipantDetectionMethodB] | None = Field(
        description='Detection method used to identify participant B in the interaction'
    )
