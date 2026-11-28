"""Interaction view: the `interaction_view` view (pts `interaction_view`).

One row per `targetA`: its interactions in `interaction` order, each with its evidence in
`interaction_evidence` order, in the served column order and types.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel
from ot_datamodels.derive import from_output
from ot_datamodels.outputs.interaction import Interaction, SpeciesA, SpeciesB
from ot_datamodels.outputs.interaction_evidence import InteractionEvidence, InteractionResources
from ot_datamodels.serving import Serving
from ot_datamodels.vocabularies import InteractionSource, SpeciesMnemonic, SpeciesScientificName

# IntAct's negative pseudo-taxids (-1 in vitro ... -4 in vivo) are served wrapped, x mod 2^32
_TaxonId = int
_UINT32_MAX = 4294967295

TargetA = from_output(Interaction, 'targetA', primary_key=True)
RowTargetA = from_output(Interaction, 'targetA')
IntA = from_output(Interaction, 'intA')
TargetB = from_output(Interaction, 'targetB')
IntB = from_output(Interaction, 'intB')
Scoring = from_output(Interaction, 'scoring')
Count = from_output(Interaction, 'count')
SourceDatabase = from_output(Interaction, 'sourceDatabase')
SpeciesATaxonId = from_output(SpeciesA, 'taxonId')
EvidenceScore = from_output(InteractionEvidence, 'evidenceScore')
ExpansionMethodMiIdentifier = from_output(InteractionEvidence, 'expansionMethodMiIdentifier')
ExpansionMethodShortName = from_output(InteractionEvidence, 'expansionMethodShortName')
HostOrganismScientificName = from_output(InteractionEvidence, 'hostOrganismScientificName')
InteractionDetectionMethodMiIdentifier = from_output(InteractionEvidence, 'interactionDetectionMethodMiIdentifier')
InteractionDetectionMethodShortName = from_output(InteractionEvidence, 'interactionDetectionMethodShortName')
InteractionIdentifier = from_output(InteractionEvidence, 'interactionIdentifier')
InteractionTypeMiIdentifier = from_output(InteractionEvidence, 'interactionTypeMiIdentifier')
InteractionTypeShortName = from_output(InteractionEvidence, 'interactionTypeShortName')
ParticipantDetectionMethodA = from_output(InteractionEvidence, 'participantDetectionMethodA')
ParticipantDetectionMethodB = from_output(InteractionEvidence, 'participantDetectionMethodB')
PubmedId = from_output(InteractionEvidence, 'pubmedId')


def _description(model: type[StructModel | DatasetModel], field: str) -> str:
    description = model.model_fields[field].description
    assert description is not None
    return description


class InteractionViewResources(StructModel):
    """`interactions[].evidences[].interactionResources`."""

    databaseVersion: str = Field(min_length=1, description=_description(InteractionResources, 'databaseVersion'))
    sourceDatabase: InteractionSource = Field(description=_description(InteractionResources, 'sourceDatabase'))


class InteractionViewEvidence(StructModel):
    """`interactions[].evidences[]`."""

    evidenceScore: EvidenceScore
    expansionMethodMiIdentifier: ExpansionMethodMiIdentifier
    expansionMethodShortName: ExpansionMethodShortName
    hostOrganismScientificName: HostOrganismScientificName
    hostOrganismTaxId: _TaxonId | None = Field(
        ge=0, le=_UINT32_MAX, description=_description(InteractionEvidence, 'hostOrganismTaxId')
    )
    interactionDetectionMethodMiIdentifier: InteractionDetectionMethodMiIdentifier
    interactionDetectionMethodShortName: InteractionDetectionMethodShortName
    interactionIdentifier: InteractionIdentifier
    interactionResources: InteractionViewResources = Field(
        description=_description(InteractionEvidence, 'interactionResources')
    )
    interactionTypeMiIdentifier: InteractionTypeMiIdentifier
    interactionTypeShortName: InteractionTypeShortName
    participantDetectionMethodA: ParticipantDetectionMethodA
    participantDetectionMethodB: ParticipantDetectionMethodB
    pubmedId: PubmedId


class InteractionViewSpeciesA(StructModel):
    """`interactions[].speciesA`."""

    mnemonic: SpeciesMnemonic = Field(description=_description(SpeciesA, 'mnemonic'))
    scientificName: SpeciesScientificName = Field(description=_description(SpeciesA, 'scientificName'))
    taxonId: SpeciesATaxonId


class InteractionViewSpeciesB(StructModel):
    """`interactions[].speciesB`."""

    mnemonic: str = Field(min_length=1, description=_description(SpeciesB, 'mnemonic'))
    scientificName: str | None = Field(min_length=1, description=_description(SpeciesB, 'scientificName'))
    taxonId: _TaxonId | None = Field(ge=0, le=_UINT32_MAX, description=_description(SpeciesB, 'taxonId'))


class InteractionViewRow(StructModel):
    """`interactions[]`."""

    targetA: RowTargetA
    intA: IntA
    targetB: TargetB
    intB: IntB
    intABiologicalRole: str = Field(min_length=1, description=_description(Interaction, 'intABiologicalRole'))
    intBBiologicalRole: str = Field(min_length=1, description=_description(Interaction, 'intBBiologicalRole'))
    scoring: Scoring
    count: Count
    sourceDatabase: SourceDatabase
    speciesA: InteractionViewSpeciesA = Field(description=_description(Interaction, 'speciesA'))
    speciesB: InteractionViewSpeciesB = Field(description=_description(Interaction, 'speciesB'))
    evidences: list[InteractionViewEvidence] = Field(min_length=1, description='Evidence supporting the interaction')


_INTERACTION = Interaction.example
_EVIDENCE = InteractionEvidence.example


class InteractionView(DatasetModel):
    """Interaction view."""

    dataset_name: ClassVar[str] = 'interaction_view'
    dataset_title: ClassVar[str] = 'Interaction view'
    dataset_description: ClassVar[str] = (
        'Molecular interactions as served, one row per target A with its interactions and their evidence nested in'
    )
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        column_types={
            'interactions[]/count': 'UInt8',
            'interactions[]/speciesA/taxonId': 'UInt32',
            'interactions[]/speciesB/taxonId': 'UInt32',
            'interactions[]/evidences[]/hostOrganismTaxId': 'UInt32',
        },
    )
    example: ClassVar[dict] = {
        'targetA': _INTERACTION['targetA'],
        'interactions': [
            {
                **{c: _INTERACTION[c] for c in InteractionViewRow.model_fields if c != 'evidences'},
                'evidences': [{c: _EVIDENCE[c] for c in InteractionViewEvidence.model_fields}],
            }
        ],
    }

    targetA: TargetA
    interactions: list[InteractionViewRow] = Field(
        min_length=1, description='Interactions of target A, in interaction dataset order'
    )
