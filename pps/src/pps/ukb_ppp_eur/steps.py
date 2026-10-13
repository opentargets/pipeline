"""UKB PPP (EUR) steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class SumstatPreprocessSettings(BaseModel):
    """Settings for `sumstat_preprocess`."""

    raw_study_index_path_from_tsv: str
    """Input raw study index path."""
    raw_summary_stats_path: str
    """Input raw summary stats path."""
    tmp_variant_annotation_path: str
    """Temporary output path for variant annotation dataset."""
    variant_annotation_path: str
    """Input variant annotation dataset path."""
    study_index_output_path: str
    """Study index output path."""
    summary_stats_output_path: str
    """Summary stats output path."""


def sumstat_preprocess(session: Session, settings: SumstatPreprocessSettings) -> None:
    """UKB PPP (EUR) data ingestion and harmonisation.

    Ported from `legacy_gentropy.ukb_ppp_eur_sumstat_preprocess.UkbPppEurStep`.
    """
    raise NotImplementedError
