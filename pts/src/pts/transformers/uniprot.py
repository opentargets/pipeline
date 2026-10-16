"""Write the parsed UniProt flat file as parquet, for the pre_target step."""

from __future__ import annotations

from typing import Any

from otter.config.model import Config

from pts.target.uniprot import read_uniprot
from pts.transformers.utils.dataset import write_dataset


def uniprot(source: str, destination: str, settings: dict[str, Any] | None, config: Config | None) -> None:
    """Parse `source` and write one row per UniProt entry to `destination`."""
    write_dataset(read_uniprot(source), destination)
