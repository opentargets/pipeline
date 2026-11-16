"""Errors that abort a staging run before it computes."""


class SeedTooSmall(Exception):  # noqa: N818
    """Raised when an imported seed covers too few of the current records."""


class ComputeTooLarge(Exception):  # noqa: N818
    """Raised when more records are left to compute than ``max_compute`` allows."""
