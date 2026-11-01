"""What every chunk of one chromosome reads, written once into the chromosome's scratch directory.

A `PreparedChromosome` names a chromosome's rows, reference contig and per-chromosome inputs,
and the files `write` prepares under its directory: its transcript fields, LOFTEE models,
ancestor contig, colocated variants, supplementary annotations and GPN-Star scores, the last
three sorted by position, and the amino-acid annotations of its transcripts' proteins. It holds
only paths, strings and numbers, so it crosses a spawn pool.
"""

from __future__ import annotations

import pickle
from dataclasses import dataclass, field
from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.variant import variant_schema
from pts.transformers.variant.alleles import read_chromosome, read_contig
from pts.transformers.variant.colocated import colocated_variants
from pts.transformers.variant.gpn_star import GpnStarCutoffs, read_gpn_star
from pts.transformers.variant.index import read_variant_annotations
from pts.transformers.variant.index_rows import uniprot_accessions
from pts.transformers.variant.loftee import loftee_models
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


def _write(frame: pl.DataFrame, path: Path, row_group_rows: int | None = None) -> Path:
    """Write Parquet through a partial file, so that a file at `path` is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f'{path.name}.partial')
    frame.write_parquet(partial, compression='zstd', row_group_size=row_group_rows)
    return partial.rename(path)


def _by_position(frame: pl.DataFrame, path: Path, row_group_rows: int) -> Path:
    """Write a prepared frame sorted by `position`, in row groups whose statistics a position range can skip."""
    return _write(frame.sort('position', maintain_order=True), path, row_group_rows)


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

    def transcripts_file(self) -> Path:
        return self._prepared / 'transcripts.parquet'

    def loftee_file(self) -> Path:
        return self._prepared / 'loftee.pickle'

    def ancestor_file(self) -> Path:
        return self._prepared / 'ancestor.bin'

    def colocated_file(self) -> Path:
        """Written only when the chromosome has `all_vars`."""
        return self._prepared / 'colocated.parquet'

    def annotations_file(self, number: int) -> Path:
        return self._prepared / f'annotations-{number:03d}.parquet'

    def gpn_star_file(self) -> Path:
        """Written only when the chromosome has GPN-Star scores."""
        return self._prepared / 'gpn_star.parquet'

    def amino_acids_file(self, number: int) -> Path:
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
        _write(transcripts, self.transcripts_file())
        self.loftee_file().write_bytes(pickle.dumps(loftee_models(sources.core, chromosome)))
        self.ancestor_file().write_bytes(_sequence(sources.ancestor, chromosome))
        if self.all_vars is not None:
            _by_position(colocated_variants(self.all_vars, frame), self.colocated_file(), row_group_rows)
        loci = frame.select('chromosome', 'position').unique().lazy()
        for number, located in enumerate(sources.variant_annotations):
            annotations = (
                read_variant_annotations(located[chromosome], self.settings.hash_threshold, loci)
                .filter(pl.col('chromosome') == chromosome)
                .collect(engine='streaming')
                if chromosome in located
                else pl.DataFrame(schema=variant_schema)
            )
            _by_position(annotations, self.annotations_file(number), row_group_rows)
        if chromosome in sources.gpn_star and sources.gpn_star_cutoffs is not None:
            scores = read_gpn_star(sources.gpn_star[chromosome], frame['position'].unique(), sources.gpn_star_cutoffs)
            _by_position(scores, self.gpn_star_file(), row_group_rows)
        for number, path in enumerate(sources.amino_acid_annotations):
            _write(amino_acid_annotations(path, transcripts), self.amino_acids_file(number))
        logger.info('Prepared chromosome {}', chromosome)
