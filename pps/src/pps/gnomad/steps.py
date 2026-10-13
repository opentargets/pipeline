"""gnomAD steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class LdIndexSettings(BaseModel):
    """Settings for `ld_index`."""

    ld_index_out: str
    """Output LD index path. (required)."""
    min_r2: float = 0.5
    """Minimum r2 to consider when considering variants within a window."""
    ld_matrix_template: str = (
        'gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.{POP}.common.adj.ld.bm'
    )
    """Input path to the gnomAD ld file with placeholder for population."""
    ld_index_raw_template: str = (
        'gs://gcp-public-data--gnomad/release/2.1.1/ld/gnomad.genomes.r2.1.1.{POP}.common.ld.variant_indices.ht'
    )
    """Input path to the raw gnomAD LD indices file with placeholder for population string."""
    liftover_ht_path: str = 'gs://gcp-public-data--gnomad/release/2.1.1/liftover_grch38/ht/genomes/gnomad.genomes.r2.1.1.sites.liftover_grch38.ht'
    """Path to the liftover ht file."""
    grch37_to_grch38_chain_path: str = 'gs://hail-common/references/grch37_to_grch38.over.chain.gz'
    """Path to the chain file used to lift over the coordinates."""
    ld_populations: list[str] = ['afr', 'amr', 'eas', 'fin', 'nfe']
    """Population names derived from the ld file paths."""


def ld_index(session: Session, settings: LdIndexSettings) -> None:
    """LD index step.

    Ported from `legacy_gentropy.gnomad_ingestion.LDIndexStep`.
    """
    raise NotImplementedError


class VariantIndexSettings(BaseModel):
    """Settings for `variant_index`."""

    variant_annotation_path: str
    """Output path for the variant annotation dataset."""
    gnomad_genomes_path: str = 'gs://gcp-public-data--gnomad/release/4.1/ht/genomes/gnomad.genomes.v4.1.sites.ht/'
    """Path to the gnomAD genomes hail table."""
    gnomad_joint_path: str = 'gs://gcp-public-data--gnomad/release/4.1/ht/joint/gnomad.joint.v4.1.sites.ht/'
    """Path to the gnomAD joint hail table."""
    gnomad_variant_populations: list[str] = ['afr', 'amr', 'ami', 'asj', 'eas', 'fin', 'nfe', 'mid', 'sas', 'remaining']
    """List of populations to include in the annotation."""


def variant_index(session: Session, settings: VariantIndexSettings) -> None:
    """A step to generate variant index dataset from gnomad data.

    Ported from `legacy_gentropy.gnomad_ingestion.GnomadVariantIndexStep`.
    """
    raise NotImplementedError
