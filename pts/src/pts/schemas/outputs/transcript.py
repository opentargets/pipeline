"""Target - transcripts: the `transcript` output.

Generated from the 26.09 release output (644,292 rows) and its croissant metadata. Rules are
strict to that release: dtypes, nullability, value ranges, allowed values and identifier shapes are as observed.
"""

from typing import Annotated, ClassVar, Literal

import polars as pl
from pydantic import Field

from pts.schemas.dataset import Bioregistry, DatasetModel, PolarsDtype, PrimaryKey, Reference, StructModel
from pts.schemas.outputs.patterns import ENSEMBL_GENE_ID, ENSEMBL_PROTEIN_ID, ENSEMBL_TRANSCRIPT_ID


class Flags(StructModel):
    """`flags[]`."""

    label: Literal['appris', 'ensembl_canonical', 'gencode_primary', 'mane'] = Field(
        description=('Category of the transcript flag (e.g. mane, appris, gencode_primary, ensembl_canonical)')
    )
    value: Literal[
        'alternative_1',
        'alternative_2',
        'alternative_M',
        'plus_clinical',
        'principal_1',
        'principal_2',
        'principal_3',
        'principal_4',
        'principal_5',
        'principal_M',
        'select',
        'true',
    ] = Field(description='Value of the transcript flag within its category')


class Exons(StructModel):
    """`exons[]`."""

    exonId: Annotated[str, Bioregistry('ensembl')] = Field(
        min_length=1, description='Ensembl identifier of the exon (ENSE###)'
    )
    chromosome: str = Field(min_length=1, description='Chromosome on which the exon is located')
    start: int = Field(ge=577, le=248936715, description='Genomic start position of the exon')
    end: int = Field(ge=647, le=248937105, description='Genomic end position of the exon')
    strand: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=-1, le=1, description=('Strand orientation of the exon: 1 for the forward strand, -1 for the reverse strand')
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
        'targetId': 'ENSG00000290825',
        'transcriptId': 'ENST00000832824',
        'biotype': 'lncRNA',
        'proteinId': 'ENSP00000493376',
        'uniprotSwissprotIds': ['Q6IEY1'],
        'uniprotTremblIds': ['A0A2U3U0J3'],
        'uniprotIsoformIds': ['Q96NU1-3'],
        'alphafoldIds': ['AF-A0A2U3U0J3-F1'],
        'isEnsemblCanonical': False,
        'chromosome': '1',
        'start': 11121,
        'end': 14413,
        'strand': 1,
        'transcriptionStartSite': 11121,
        'flags': [{'label': 'gencode_primary', 'value': 'true'}],
        'exons': [{'exonId': 'ENSE00004248723', 'chromosome': '1', 'start': 11121, 'end': 11211, 'strand': 1}],
    }

    targetId: Annotated[str, Bioregistry('ensembl'), Reference('target', 'id')] = Field(
        pattern=ENSEMBL_GENE_ID, description='Ensembl gene identifier of the target the transcript belongs to'
    )
    transcriptId: Annotated[str, Bioregistry('ensembl'), PrimaryKey()] = Field(
        pattern=ENSEMBL_TRANSCRIPT_ID, description='Ensembl identifier of the transcript (ENST###)'
    )
    biotype: str = Field(min_length=1, description='Biotype classification of the transcript')
    proteinId: Annotated[str, Bioregistry('ensembl')] | None = Field(
        pattern=ENSEMBL_PROTEIN_ID,
        description=('Ensembl protein identifier for the transcript (ENSP###); null for non-coding transcripts'),
    )
    uniprotSwissprotIds: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] | None = Field(
        min_length=1,
        description=('Reviewed (Swiss-Prot) UniProt identifiers for the protein translated from the transcript'),
    )
    uniprotTremblIds: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] | None = Field(
        min_length=1,
        description=('Unreviewed (TrEMBL) UniProt identifiers for the protein translated from the transcript'),
    )
    uniprotIsoformIds: list[Annotated[Annotated[str, Bioregistry('uniprot')], Field(min_length=1)]] | None = Field(
        min_length=1, description=('UniProt isoform identifiers for the protein translated from the transcript')
    )
    alphafoldIds: list[Annotated[str, Field(min_length=1)]] | None = Field(
        min_length=1, description=('AlphaFold structure identifiers for the protein translated from the transcript')
    )
    isEnsemblCanonical: bool = Field(
        description=('Flag indicating if the transcript is the Ensembl canonical transcript for the gene')
    )
    chromosome: str = Field(min_length=1, description='Chromosome on which the transcript is located')
    start: int = Field(ge=577, le=248936581, description='Genomic start position of the transcript')
    end: int = Field(ge=647, le=248937105, description='Genomic end position of the transcript')
    strand: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=-1,
        le=1,
        description=('Strand orientation of the transcript: 1 for the forward strand, -1 for the reverse strand'),
    )
    transcriptionStartSite: Annotated[int, PolarsDtype(pl.Int32)] = Field(
        ge=577,
        le=248936913,
        description=(
            "Transcription Start Site (TSS) genomic position, derived from the transcript's strand and coordinates"
        ),
    )
    flags: list[Flags] = Field(
        description=(
            'List of curation flags describing the transcript (e.g. MANE, APPRIS, GENCODE Primary, Ensembl canonical)'
        )
    )
    exons: list[Exons] = Field(min_length=1, description='List of exons that make up the transcript')
