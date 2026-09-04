"""Search index generation for diseases, targets, drugs, variants and studies."""

from __future__ import annotations

from typing import Any

from otter.config.model import Config


def search(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Placeholder; implemented in Task 10."""
    raise NotImplementedError
