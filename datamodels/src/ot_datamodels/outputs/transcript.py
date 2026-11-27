"""Target - transcripts: the `transcript` output.

Generated from the 26.12 release output (644,292 rows) and its croissant metadata.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from ot_datamodels import Bioregistry, ClickHouseType, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from ot_datamodels.outputs.patterns import ENSEMBL_GENE_ID, ENSEMBL_PROTEIN_ID, ENSEMBL_TRANSCRIPT_ID
from ot_datamodels.vocabularies import Chromosome, TranscriptBiotype, TranscriptFlagLabel, TranscriptFlagValue


class DbXrefs(StructModel):
    """`dbXrefs[]`."""

    id: str = Field(min_length=1, description='Cross-reference ID for external databases')
    source: Literal['RNAcentral'] = Field(description='Source database for the cross-reference')


class Flags(StructModel):
    """`flags[]`."""

    label: Annotated[TranscriptFlagLabel, PolarsDtype(pl.Enum)] = Field(
        description=('Category of the transcript flag (e.g. mane, appris, gencode_primary, ensembl_canonical)')
    )
    value: Annotated[TranscriptFlagValue, PolarsDtype(pl.Enum)] = Field(
        description='Value of the transcript flag within its category'
    )


class Exons(StructModel):
    """`exons[]`."""

    exonId: Annotated[str, Bioregistry('ensembl')] = Field(
        min_length=1, description='Ensembl identifier of the exon (ENSE###)'
    )
    chromosome: Annotated[Chromosome, PolarsDtype(pl.Enum)] = Field(
        description='Chromosome on which the exon is located'
    )
    start: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='Genomic start position of the exon'
    )
    end: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='Genomic end position of the exon'
    )
    strand: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType("Enum8('-1' = -1, '0' = 0, '1' = 1)")] = Field(
        description=('Strand orientation of the exon: 1 for the forward strand, -1 for the reverse strand')
    )


class Transcript(DatasetModel):
    """Target - transcripts."""

    dataset_name: ClassVar[str] = 'transcript'
    dataset_title: ClassVar[str] = 'Target - transcripts'
    dataset_description: ClassVar[str] = (
        'Per-transcript index derived from the Ensembl core database, '
        'including genomic coordinates, biotype, protein/UniProt identifiers, '
        'and curation flags (MANE, APPRIS, GENCODE Primary, Ensembl canonical)'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Target',)
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000000003',
        'transcriptId': 'ENST00000373020',
        'biotype': 'protein_coding',
        'proteinId': 'ENSP00000362111',
        'uniprotSwissprotIds': ['O43657'],
        'uniprotTremblIds': ['A0A087WYV6'],
        'uniprotIsoformIds': ['Q9H2S6-1'],
        'alphafoldIds': ['AF-O43657-F1'],
        'dbXrefs': [{'id': 'URS0000E9BD10_9606', 'source': 'RNAcentral'}],
        'isEnsemblCanonical': True,
        'chromosome': 'X',
        'start': 100627108,
        'end': 100636806,
        'strand': -1,
        'transcriptionStartSite': 100636806,
        'flags': [{'label': 'mane', 'value': 'select'}],
        'exons': [{'exonId': 'ENSE00001855382', 'chromosome': 'X', 'start': 100636608, 'end': 100636806, 'strand': -1}],
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target the transcript belongs to'
    )
    transcriptId: Annotated[str, Bioregistry('ensembl'), PrimaryKey()] = Field(
        pattern=ENSEMBL_TRANSCRIPT_ID, description='Ensembl identifier of the transcript (ENST###)'
    )
    biotype: Annotated[TranscriptBiotype, PolarsDtype(pl.Enum)] = Field(
        description='Biotype classification of the transcript'
    )
    proteinId: Annotated[str, Bioregistry('ensembl')] | None = Field(
        pattern=ENSEMBL_PROTEIN_ID,
        description=('Ensembl protein identifier for the transcript (ENSP###); null for non-coding transcripts'),
    )
    uniprotSwissprotIds: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] | None = Field(
        description=('Reviewed (Swiss-Prot) UniProt identifiers for the protein translated from the transcript')
    )
    uniprotTremblIds: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] | None = Field(
        description=('Unreviewed (TrEMBL) UniProt identifiers for the protein translated from the transcript')
    )
    uniprotIsoformIds: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] | None = Field(
        description=('UniProt isoform identifiers for the protein translated from the transcript')
    )
    alphafoldIds: list[Annotated[str, Field(min_length=1)]] | None = Field(
        description=('AlphaFold structure identifiers for the protein translated from the transcript')
    )
    dbXrefs: list[DbXrefs] | None = Field(description='Database cross-references for the transcript')
    isEnsemblCanonical: bool = Field(
        description=('Flag indicating if the transcript is the Ensembl canonical transcript for the gene')
    )
    chromosome: Annotated[Chromosome, PolarsDtype(pl.Enum)] = Field(
        description='Chromosome on which the transcript is located'
    )
    start: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='Genomic start position of the transcript'
    )
    end: Annotated[int, ClickHouseType('UInt32')] = Field(
        ge=0, le=4294967295, description='Genomic end position of the transcript'
    )
    strand: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType("Enum8('-1' = -1, '0' = 0, '1' = 1)")] = Field(
        description=('Strand orientation of the transcript: 1 for the forward strand, -1 for the reverse strand')
    )
    transcriptionStartSite: Annotated[int, PolarsDtype(pl.Int32), ClickHouseType('UInt32')] = Field(
        ge=0,
        description=(
            "Transcription Start Site (TSS) genomic position, derived from the transcript's strand and coordinates"
        ),
    )
    flags: list[Flags] = Field(
        description=(
            'List of curation flags describing the transcript (e.g. MANE, APPRIS, GENCODE Primary, Ensembl canonical)'
        )
    )
    exons: list[Exons] = Field(description='List of exons that make up the transcript')
