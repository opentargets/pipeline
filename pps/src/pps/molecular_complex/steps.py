"""Molecular complex steps."""

from __future__ import annotations

from pydantic import BaseModel

from pps.session import Session


class IngestionSettings(BaseModel):
    """Settings for `ingestion`."""

    predicted_complex_tab_path: str
    """Path to the predicted protein-complex tab-separated file."""
    experimental_complex_tab_path: str
    """Path to the experimental protein-complex tab-separated file."""
    output_path: str
    """Destination path for the merged `MolecularComplex` Parquet dataset."""


def ingestion(session: Session, settings: IngestionSettings) -> None:
    """Ingest predicted and experimental protein-complex data into a `MolecularComplex` Parquet dataset.

    Ported from `legacy_gentropy.molecular_complex.MolecularComplexIngestionStep`.
    """
    raise NotImplementedError
