"""GERP conservation: the highest full-resolution score over each variant's span."""

from __future__ import annotations

import numpy as np
import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs

WINDOW_GAP = 1 << 12
WINDOW_BASES = 1 << 22


def _windows(low: np.ndarray, high: np.ndarray, gap: int, longest: int) -> list[tuple[int, int, np.ndarray]]:
    """Merge spans, sorted by `low`, into read windows: (start, end, indices of the spans in it)."""
    windows: list[tuple[int, int, np.ndarray]] = []
    first = 0
    window_low, window_high = int(low[0]), int(high[0])
    for i in range(1, low.size):
        if low[i] > window_high + gap or max(window_high, high[i]) - window_low > longest:
            windows.append((window_low, window_high, np.arange(first, i)))
            first, window_low, window_high = i, int(low[i]), int(high[i])
        else:
            window_high = max(window_high, int(high[i]))
    windows.append((window_low, window_high, np.arange(first, low.size)))
    return windows


def gerp_max(
    bigwig: str, chromosome: str, spans: pl.DataFrame, gap: int = WINDOW_GAP, longest: int = WINDOW_BASES
) -> pl.Series:
    """The highest GERP score of each span, over the bigWig's full-resolution values.

    An insertion (it ends before it starts) takes the two flanking bases. Spans are read in
    windows: sorted spans less than `gap` bases apart share one read of at most `longest` bases,
    and each span's maximum is taken over its bases of the window.

    Args:
        bigwig: The GERP bigWig, a path pyBigWig can open.
        chromosome: The contig name as the bigWig spells it.
        spans: One row per span, with VEP's one-based `vepStart` and `vepEnd`.
        gap: Bases between spans below which they share a read.
        longest: Bases a read covers at most, beyond those of its longest span.

    Returns:
        Float64, the numeric maximum, null where no base of the span is covered or the bigWig
        has no such contig.
    """
    starts = spans['vepStart'].cast(pl.Int64).to_numpy()
    ends = spans['vepEnd'].cast(pl.Int64).to_numpy()
    scores = np.full(starts.size, np.nan)
    with pyBigWig.open(bigwig) as handle:
        length = (handle.chroms() or {}).get(chromosome)
        insertion = starts - 1 == ends
        low = np.maximum(np.where(insertion, starts - 2, starts - 1), 0)
        high = np.minimum(np.where(insertion, ends + 1, ends), length or 0)
        read = np.flatnonzero(low < high) if length is not None else np.zeros(0, dtype=np.int64)
        if read.size:
            read = read[np.argsort(low[read], kind='stable')]
            for window_low, window_high, members in _windows(low[read], high[read], gap, longest):
                values = np.append(handle.values(chromosome, window_low, window_high, numpy=True), np.nan)
                index = read[members]
                bounds = np.column_stack([low[index], high[index]]).ravel() - window_low
                scores[index] = np.fmax.reduceat(values, bounds)[::2]
    return pl.Series('gerp', scores, dtype=pl.Float64).fill_nan(None)
