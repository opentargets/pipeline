"""Transcript view: the `transcript_view` view (pts `target_view`).

One row per target with its `transcript` rows nested in, in the transcript's column order.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.transcript import Transcript
from ot_datamodels.serving import Serving
from ot_datamodels.vocabularies import TranscriptFlagLabel, TranscriptFlagValue

TargetId = from_output(Transcript, 'targetId', primary_key=True)
TranscriptId = from_output(Transcript, 'transcriptId')
ProteinId = from_output(Transcript, 'proteinId')
UniprotSwissprotIds = from_output(Transcript, 'uniprotSwissprotIds')
UniprotTremblIds = from_output(Transcript, 'uniprotTremblIds')
UniprotIsoformIds = from_output(Transcript, 'uniprotIsoformIds')
AlphafoldIds = from_output(Transcript, 'alphafoldIds')
DbXrefs = from_output(Transcript, 'dbXrefs')
IsEnsemblCanonical = from_output(Transcript, 'isEnsemblCanonical')
Chromosome = from_output(Transcript, 'chromosome')
Start = from_output(Transcript, 'start')
End = from_output(Transcript, 'end')
Strand = from_output(Transcript, 'strand')
TranscriptionStartSite = from_output(Transcript, 'transcriptionStartSite')
Exons = from_output(Transcript, 'exons')


class TranscriptFlag(StructModel):
    """`transcripts[].flags[]`: a transcript flag, its Enums served as nullable strings."""

    label: TranscriptFlagLabel | None = Field(description='Category of the transcript flag')
    value: TranscriptFlagValue | None = Field(description='Value of the transcript flag within its category')


class TranscriptViewRow(StructModel):
    """`transcripts[]`: a `transcript` row without its `targetId`, `biotype` without LowCardinality."""

    transcriptId: TranscriptId
    biotype: str = Field(min_length=1, description='Biotype classification of the transcript')
    proteinId: ProteinId
    uniprotSwissprotIds: UniprotSwissprotIds
    uniprotTremblIds: UniprotTremblIds
    uniprotIsoformIds: UniprotIsoformIds
    alphafoldIds: AlphafoldIds
    dbXrefs: DbXrefs
    isEnsemblCanonical: IsEnsemblCanonical
    chromosome: Chromosome
    start: Start
    end: End
    strand: Strand
    transcriptionStartSite: TranscriptionStartSite
    flags: list[TranscriptFlag] = Field(description='Curation flags of the transcript')
    exons: Exons


class TranscriptView(DatasetModel):
    """Transcript view."""

    dataset_name: ClassVar[str] = 'transcript_view'
    dataset_title: ClassVar[str] = 'Transcript view'
    dataset_description: ClassVar[str] = 'The transcripts of each target, by transcriptId'
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = {
        'targetId': Transcript.example['targetId'],
        'transcripts': [{k: v for k, v in Transcript.example.items() if k != 'targetId'}],
    }

    targetId: TargetId
    transcripts: list[TranscriptViewRow] = Field(min_length=1, description='Transcripts of the target')
