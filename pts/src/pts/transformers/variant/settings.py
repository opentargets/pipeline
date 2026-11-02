"""The source, destination and settings of the variant step, as `config.yaml` gives them."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.variant.annotators.gpn_star import GpnStarCutoffs


class VariantSource(BaseModel):
    """Variant rows, the reference data fastVEP and its joins read, and supplementary annotation paths."""

    model_config = ConfigDict(extra='forbid')

    uniprot_variants: str
    eva: str
    pharmacogenomics: str
    credible_set: str
    fasta: str
    """The gzip GRCh38 primary assembly FASTA."""
    gff3: str
    gerp: str
    """The GERP conservation bigWig."""
    vep_cache: str
    """The indexed VEP cache tarball, read for its `all_vars.gz`."""
    sift_predictions: str
    """The variation dump's `protein_function_predictions.txt.gz`."""
    sift_translation_md5: str
    sift_attrib: str
    peptides: str
    """The `pep.all.fa.gz` FASTA, whose md5s key the SIFT matrices."""
    alphamissense: str
    ancestor: str
    """The gzip LOFTEE human ancestor FASTA."""
    core: str
    """Directory of the Ensembl core database dump."""
    variant_annotations: list[str] = Field(default_factory=list)
    amino_acid_annotations: list[str] = Field(default_factory=list)
    gpn_star: str | None = None


class VariantDestination(BaseModel):
    """Variant index output path."""

    model_config = ConfigDict(extra='forbid')

    output: str


class VariantSettings(BaseModel):
    """Identifier hashing, fastVEP and parallelism."""

    model_config = ConfigDict(extra='forbid')

    hash_threshold: int = Field(default=300, gt=0)
    """Identifiers at least this long are hashed."""
    fastvep_binary: str = 'fastvep'
    fastvep_threads: int = Field(default=8, gt=0)
    """Threads of each fastVEP run."""
    parallel_chromosomes: int = Field(default=4, gt=0)
    """Chromosomes run through fastVEP at the same time."""
    parallel_preparations: int = Field(default=4, gt=0)
    """Chromosomes whose shared chunk inputs are prepared at the same time."""
    chromosome_threads: int = Field(default=8, gt=0)
    """Polars threads of each fastVEP and preparation process."""
    chunk_rows: int = Field(default=50_000, gt=0)
    """Rows of each chunk a chromosome is annotated in after fastVEP."""
    parallel_chunks: int = Field(default=16, gt=0)
    """Chunks annotated at the same time."""
    chunk_threads: int = Field(default=2, gt=0)
    """Polars threads of each chunk process."""
    preparation_niceness: int = Field(default=5, ge=0, le=19)
    """Niceness of each preparation process; fastVEP keeps the step's own."""
    chunk_niceness: int = Field(default=15, ge=0, le=19)
    """Niceness of each chunk process."""
    batch_size: int = Field(default=5_000, gt=0)
    """Variants per batch when merging the annotations into a chunk."""
    gpn_star: GpnStarCutoffs | None = None
    """Cut-offs normalising GPN-Star scores; required with a GPN-Star source."""
