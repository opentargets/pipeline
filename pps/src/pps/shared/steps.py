"""Steps shared across datasources."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel

from pps.session import Session


class LdClumpingSettings(BaseModel):
    """Settings for `ld_clumping`."""

    study_locus_input_path: str
    """Path to the input study locus."""
    study_index_path: str
    """Path to the study index."""
    ld_index_path: str
    """Path to the LD index."""
    clumped_study_locus_output_path: str
    """Output path for the LD-clumped study locus."""


def ld_clumping(session: Session, settings: LdClumpingSettings) -> None:
    """Perform LD-based clumping on a study locus dataset.

    Annotates the study loci with population-specific linked variants from the LD
    index (weighted by the ancestries in the study index), then flags study loci that
    are explained by a more significant association from the same study.

    Ported from `legacy_gentropy.ld_based_clumping.LDBasedClumpingStep`.
    """
    raise NotImplementedError


class PicsSettings(BaseModel):
    """Settings for `pics`."""

    study_locus_ld_annotated_in: str
    """Path to the LD-annotated study locus."""
    picsed_study_locus_out: str
    """Output path for the PICS credible sets."""
    credible_interval: Literal['IS95', 'IS99'] = 'IS99'
    """Credible interval the credible sets are filtered to."""
    pvalue_cutoff: float = 1e-8
    """Lead p-value below which loci are kept unflagged."""


def pics(session: Session, settings: PicsSettings) -> None:
    """Fine-map an LD-annotated study locus dataset with PICS.

    Builds credible sets with PICS, filters them to `credible_interval` and flags
    loci whose lead p-value is above `pvalue_cutoff`.

    Ported from `legacy_gentropy.pics.PICSStep`.
    """
    raise NotImplementedError
