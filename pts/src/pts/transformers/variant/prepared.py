"""What every chunk of one chromosome reads, written once into the chromosome's scratch directory.

A `PreparedChromosome` names a chromosome's rows, reference contig and per-chromosome inputs,
and the files `write` prepares under its directory: its transcript fields, LOFTEE models,
ancestor contig, colocated variants, supplementary annotations and GPN-Star scores, the last
three sorted by position, and the amino-acid annotations of its transcripts' proteins. It holds
only paths, strings and numbers, so it crosses a spawn pool. Its `chunk` reads one chunk's
`ChunkInputs` from those files, the position-sorted ones only in the chunk's position range.
"""

from __future__ import annotations

import pickle
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import polars as pl
from loguru import logger

from pts.schemas.variant import variant_schema
from pts.transformers.variant.alleles import read_chromosome, read_contig
from pts.transformers.variant.colocated import KEY, colocated_variants
from pts.transformers.variant.gpn_star import GpnStarCutoffs, read_gpn_star
from pts.transformers.variant.index import read_variant_annotations
from pts.transformers.variant.index_rows import uniprot_accessions
from pts.transformers.variant.loftee import TranscriptModel, loftee_models
from pts.transformers.variant.transcripts import transcript_fields

AMINO_ACID_COLUMNS = ('uniprotAccession', 'aminoAcidChange', 'variantEffect')


@dataclass(frozen=True)
class ChunkSettings:
    """How a chromosome is cut into chunks and how each chunk is annotated."""

    chunk_rows: int = 50_000
    """Rows of each chunk of a chromosome."""
    batch_size: int = 5_000
    """Variants per batch when merging the annotations into a chunk."""
    row_group_rows: int = 10_000
    """Rows of each row group of the prepared frames, the unit a chunk's position range reads."""
    hash_threshold: int = 300
    """Identifiers at least this long are hashed."""


@dataclass(frozen=True)
class PreparationSources:
    """The local inputs every chromosome's preparation reads."""

    ancestor: str
    """The local gzip human ancestor FASTA."""
    core: str
    """Directory of the local Ensembl core database dump."""
    variant_annotations: tuple[dict[str, list[str]], ...] = ()
    """The Parquet parts of each supplementary variant annotation source, by chromosome."""
    amino_acid_annotations: tuple[Path, ...] = ()
    """The local `amino_acid_projection` of each amino-acid annotation source (FoldX)."""
    gpn_star: dict[str, str] = field(default_factory=dict)
    """The GPN-Star file of each chromosome that has one."""
    gpn_star_cutoffs: GpnStarCutoffs | None = None


def _sequence(fasta: str, chromosome: str) -> bytes:
    """The chromosome's contig of a gzip FASTA, empty for a contig the FASTA lacks."""
    try:
        return read_chromosome(fasta, chromosome)
    except ValueError:
        return b''


def write_complete(frame: pl.DataFrame, path: Path, row_group_rows: int | None = None) -> Path:
    """Write Parquet through a partial file, so that a file at `path` is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'{path.name}.partial')
    frame.write_parquet(partial, compression='zstd', row_group_size=row_group_rows)
    return partial.rename(path)


def _by_position(frame: pl.DataFrame, path: Path, row_group_rows: int) -> Path:
    """Write a prepared frame sorted by `position`, in row groups whose statistics a position range can skip."""
    return write_complete(frame.sort('position', maintain_order=True), path, row_group_rows)


def in_range(path: Path, positions: pl.Series) -> pl.LazyFrame:
    """The rows of a Parquet file with `position` between the least and the greatest of `positions`."""
    return pl.scan_parquet(path).filter(pl.col('position').is_between(positions.min(), positions.max()))


def amino_acid_annotations(path: Path, transcripts: pl.DataFrame) -> pl.DataFrame:
    """The rows of an amino-acid annotation file on a UniProt accession of one of `transcripts`."""
    accessions = (
        transcripts
        .select(uniprot_accessions(pl.col('swissprot'), pl.col('trembl')).alias('uniprotAccession'))
        .explode('uniprotAccession')
        .drop_nulls()
        .unique()
    )
    return (
        pl
        .scan_parquet(path)
        .select(AMINO_ACID_COLUMNS)
        .join(accessions.lazy(), on='uniprotAccession', how='semi', maintain_order='left')
        .collect()
    )


@dataclass(frozen=True)
class PreparedChromosome:
    """One chromosome's rows and inputs, and the files prepared for its chunks under `directory`."""

    chromosome: str
    directory: Path
    """The chromosome's scratch directory, holding its fastVEP run, records and preparation."""
    rows: Path
    """Parquet of the chromosome's rows, sorted by position."""
    plain_fasta: Path
    """The uncompressed reference, which fastVEP and the chunks read."""
    span: tuple[int, int] | None
    """The chromosome's `contig_spans` span of `plain_fasta`, None for a contig it lacks."""
    all_vars: Path | None
    """The chromosome's `all_vars.gz`, None when the cache has none."""
    alphamissense: str
    """The chromosome's `partition_alphamissense` part."""
    gerp: str
    """The local GERP bigWig."""
    sift: Path
    """`sift_matrices`, as Parquet."""
    settings: ChunkSettings
    annotation_sources: int
    """The number of supplementary variant annotation sources."""
    amino_acid_sources: int
    """The number of amino-acid annotation sources."""

    def sequence(self) -> bytes:
        """The chromosome's contig of the uncompressed reference, empty for a contig it lacks."""
        return b'' if self.span is None else read_contig(self.plain_fasta, self.span)

    def records(self, chunk: int) -> Path:
        """The directory of one chunk's fastVEP records."""
        return self.directory / 'records' / f'{chunk:05d}'

    def record_slice(self, chunk: int, number: int) -> Path:
        """The file of one chunk's records from the `number`-th slice of the fastVEP output."""
        return self.records(chunk) / f'{number:05d}.parquet'

    @property
    def _prepared(self) -> Path:
        return self.directory / 'prepared'

    def _transcripts(self) -> Path:
        return self._prepared / 'transcripts.parquet'

    def _loftee(self) -> Path:
        return self._prepared / 'loftee.pickle'

    def _ancestor(self) -> Path:
        return self._prepared / 'ancestor.bin'

    def _colocated(self) -> Path:
        """Written only when the chromosome has `all_vars`."""
        return self._prepared / 'colocated.parquet'

    def _annotations(self, number: int) -> Path:
        return self._prepared / f'annotations-{number:03d}.parquet'

    def _gpn_star(self) -> Path:
        """Written only when the chromosome has GPN-Star scores."""
        return self._prepared / 'gpn_star.parquet'

    def _amino_acids(self, number: int) -> Path:
        return self._prepared / f'amino-acids-{number:03d}.parquet'

    def write(self, sources: PreparationSources) -> None:
        """Write what every chunk of the chromosome reads, once, into its `prepared` directory.

        That is its transcript fields, LOFTEE models and ancestor contig, its rows' colocated
        variants, the supplementary annotations and GPN-Star scores at its rows' loci, the last
        three sorted by position, and the amino-acid annotations of its transcripts' proteins.

        Raises:
            ValueError: if `sources` has a number of annotation sources other than the handle's.
            FileExistsError: if the chromosome is already, or partly, prepared.
        """
        counts = (len(sources.variant_annotations), len(sources.amino_acid_annotations))
        if counts != (self.annotation_sources, self.amino_acid_sources):
            raise ValueError(
                f'chromosome {self.chromosome}: {counts} variant and amino-acid annotation sources for a handle '
                f'of {(self.annotation_sources, self.amino_acid_sources)}'
            )
        chromosome, row_group_rows = self.chromosome, self.settings.row_group_rows
        frame = pl.read_parquet(self.rows)
        self._prepared.mkdir(parents=True)
        transcripts = transcript_fields(sources.core, chromosome)
        write_complete(transcripts, self._transcripts())
        self._loftee().write_bytes(pickle.dumps(loftee_models(sources.core, chromosome)))
        self._ancestor().write_bytes(_sequence(sources.ancestor, chromosome))
        if self.all_vars is not None:
            _by_position(colocated_variants(self.all_vars, frame), self._colocated(), row_group_rows)
        loci = frame.select('chromosome', 'position').unique().lazy()
        for number, located in enumerate(sources.variant_annotations):
            annotations = (
                read_variant_annotations(located[chromosome], self.settings.hash_threshold, loci)
                .filter(pl.col('chromosome') == chromosome)
                .collect(engine='streaming')
                if chromosome in located
                else pl.DataFrame(schema=variant_schema)
            )
            _by_position(annotations, self._annotations(number), row_group_rows)
        if chromosome in sources.gpn_star and sources.gpn_star_cutoffs is not None:
            scores = read_gpn_star(sources.gpn_star[chromosome], frame['position'].unique(), sources.gpn_star_cutoffs)
            _by_position(scores, self._gpn_star(), row_group_rows)
        for number, path in enumerate(sources.amino_acid_annotations):
            write_complete(amino_acid_annotations(path, transcripts), self._amino_acids(number))
        logger.info('Prepared chromosome {}', chromosome)

    def chunk(self, chunk: int, annotated: bool) -> PreparedChunk:
        """The inputs of one chunk: `chunk_rows` rows from row `chunk * chunk_rows`.

        Args:
            chunk: The chunk.
            annotated: Whether fastVEP ran on the chromosome. Without it, the chunk has no records,
                transcripts, LOFTEE models or ancestor, and none of their files is read.
        """
        offset = chunk * self.settings.chunk_rows
        rows = pl.scan_parquet(self.rows).slice(offset, self.settings.chunk_rows).collect()
        records = transcripts = None
        models: dict[str, TranscriptModel] = {}
        ancestor = b''
        if annotated:
            records = pl.scan_parquet(self.records(chunk) / '*.parquet').with_columns(pl.col('row') - offset).collect()
            transcripts = pl.read_parquet(self._transcripts())
            models = pickle.loads(self._loftee().read_bytes())
            ancestor = self._ancestor().read_bytes()
        located = self._colocated()
        colocated = (
            in_range(located, rows['position']).join(rows.lazy().select(KEY), on=KEY, how='semi').collect()
            if located.exists()
            else None
        )
        scores = self._gpn_star()
        return PreparedChunk(
            chromosome=self.chromosome,
            rows=rows,
            sequence=self.sequence(),
            records=records,
            transcripts=transcripts,
            models=models,
            ancestor=ancestor,
            gerp=self.gerp,
            alphamissense=self.alphamissense,
            colocated=colocated,
            sift=self.sift,
            annotation_files=tuple(self._annotations(number) for number in range(self.annotation_sources)),
            gpn_star_file=scores if scores.exists() else None,
            amino_acid_files=tuple(self._amino_acids(number) for number in range(self.amino_acid_sources)),
        )

    def release(self, chunk: int) -> None:
        """Delete one chunk's fastVEP records."""
        shutil.rmtree(self.records(chunk))


class ChunkInputs(Protocol):
    """What the annotation of one chunk of a chromosome's rows reads."""

    @property
    def chromosome(self) -> str:
        """The chromosome of the rows."""
        ...

    @property
    def rows(self) -> pl.DataFrame:
        """The chunk's rows."""
        ...

    @property
    def sequence(self) -> bytes:
        """The chromosome's reference, empty for a contig the reference lacks."""
        ...

    @property
    def records(self) -> pl.DataFrame | None:
        """The chunk's fastVEP records, `row` the ordinal in `rows`; None, as `transcripts`, without fastVEP."""
        ...

    @property
    def transcripts(self) -> pl.DataFrame | None:
        """The chromosome's `transcript_fields`."""
        ...

    @property
    def models(self) -> dict[str, TranscriptModel]:
        """The chromosome's `loftee_models`, empty without fastVEP."""
        ...

    @property
    def ancestor(self) -> bytes:
        """The chromosome's human ancestor, empty without fastVEP."""
        ...

    @property
    def gerp(self) -> str:
        """The GERP bigWig."""
        ...

    @property
    def alphamissense(self) -> str:
        """The chromosome's AlphaMissense predictions."""
        ...

    @property
    def colocated(self) -> pl.DataFrame | None:
        """The colocated variants of the chunk's rows, None without `all_vars`."""
        ...

    def sift_matrices(self, transcript_ids: pl.Series) -> pl.DataFrame:
        """The SIFT matrices of the transcripts."""
        ...

    def annotations(self, loci: pl.DataFrame) -> list[pl.DataFrame]:
        """Each supplementary variant annotation source's rows at the `chromosome`, `position` loci."""
        ...

    def gpn_star(self, positions: pl.Series) -> pl.DataFrame | None:
        """The GPN-Star scores at the positions, None for a chromosome without GPN-Star scores."""
        ...

    def amino_acids(self, keys: pl.DataFrame) -> list[pl.DataFrame]:
        """Each amino-acid annotation source's rows on the `uniprotAccession`, `aminoAcidChange` keys."""
        ...


@dataclass(frozen=True)
class PreparedChunk:
    """The `ChunkInputs` of one chunk, read from its chromosome's prepared files."""

    chromosome: str
    rows: pl.DataFrame
    sequence: bytes
    records: pl.DataFrame | None
    transcripts: pl.DataFrame | None
    models: dict[str, TranscriptModel]
    ancestor: bytes
    gerp: str
    alphamissense: str
    colocated: pl.DataFrame | None
    sift: Path
    annotation_files: tuple[Path, ...]
    gpn_star_file: Path | None
    amino_acid_files: tuple[Path, ...]

    def sift_matrices(self, transcript_ids: pl.Series) -> pl.DataFrame:
        wanted = transcript_ids.to_frame('transcript_id').unique()
        return pl.scan_parquet(self.sift).join(wanted.lazy(), on='transcript_id', how='semi').collect()

    def annotations(self, loci: pl.DataFrame) -> list[pl.DataFrame]:
        return [
            in_range(path, loci['position']).join(loci.lazy(), on=['chromosome', 'position'], how='semi').collect()
            for path in self.annotation_files
        ]

    def gpn_star(self, positions: pl.Series) -> pl.DataFrame | None:
        if self.gpn_star_file is None:
            return None
        return in_range(self.gpn_star_file, positions).filter(pl.col('position').is_in(positions.implode())).collect()

    def amino_acids(self, keys: pl.DataFrame) -> list[pl.DataFrame]:
        return [
            pl.scan_parquet(path).join(keys.lazy(), on=['uniprotAccession', 'aminoAcidChange'], how='semi').collect()
            for path in self.amino_acid_files
        ]
