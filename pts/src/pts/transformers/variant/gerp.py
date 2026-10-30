"""GERP conservation read from the bigWig, and the kent bigWig arithmetic LOFTEE depends on."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

import numpy as np
import polars as pl
import pyBigWig  # ty: ignore[unresolved-import] -- compiled module without stubs


class KentBigWig:
    """A bigWig read the way kent's `bigWigSummaryArrayExtended` (v335) reads it."""

    def __init__(self, path: str):
        self.path = path
        self.bw = pyBigWig.open(path)
        self.f = Path(path).open('rb')  # noqa: SIM115
        head = self._read(0, 64)
        (_, _, n_zoom, self.chrom_tree, _, _, _, _, _, _, self.buf_size, _) = struct.unpack('<IHHQQQHHQQIQ', head)
        self.zooms = []
        for i in range(n_zoom):
            reduction, _, _, index_off = struct.unpack('<IIQQ', self._read(64 + 24 * i, 24))
            self.zooms.append((reduction, index_off))
        self.chrom_ids = self._chrom_ids()
        self._zoom_cache: dict[tuple[int, str], tuple[np.ndarray, ...]] = {}

    def close(self) -> None:
        """Close both views of the file."""
        self.bw.close()
        self.f.close()

    def __enter__(self) -> KentBigWig:
        return self

    def __exit__(self, *_) -> None:
        self.close()

    def _read(self, off: int, n: int) -> bytes:
        self.f.seek(off)
        return self.f.read(n)

    def _chrom_ids(self) -> dict[str, int]:
        _, _, key_size, _, _, _ = struct.unpack('<IIIIQQ', self._read(self.chrom_tree, 32))
        out = {}

        def node(off):
            is_leaf, _, count = struct.unpack('<BBH', self._read(off, 4))
            pos = off + 4
            for _ in range(count):
                key = self._read(pos, key_size).rstrip(b'\0').decode()
                if is_leaf:
                    cid, _ = struct.unpack('<II', self._read(pos + key_size, 8))
                    out[key] = cid
                    pos += key_size + 8
                else:
                    (child,) = struct.unpack('<Q', self._read(pos + key_size, 8))
                    node(child)
                    pos += key_size + 8

        node(self.chrom_tree + 32)
        return out

    def _zoom_records(self, level: int, chrom: str):
        key = (level, chrom)
        if key in self._zoom_cache:
            return self._zoom_cache[key]
        cid = self.chrom_ids[chrom]
        index_off = self.zooms[level][1]
        blocks = []

        def node(off):
            is_leaf, _, count = struct.unpack('<BBH', self._read(off, 4))
            pos = off + 4
            for _ in range(count):
                if is_leaf:
                    sc, _, ec, _, doff, dsize = struct.unpack('<IIIIQQ', self._read(pos, 32))
                    if sc <= cid <= ec:
                        blocks.append((doff, dsize))
                    pos += 32
                else:
                    sc, _, ec, _, child = struct.unpack('<IIIIQ', self._read(pos, 24))
                    if sc <= cid <= ec:
                        node(child)
                    pos += 24

        node(index_off + 48)
        dt = np.dtype([
            ('chrom', '<u4'),
            ('start', '<u4'),
            ('end', '<u4'),
            ('valid', '<u4'),
            ('min', '<f4'),
            ('max', '<f4'),
            ('sum', '<f4'),
            ('sumsq', '<f4'),
        ])
        parts = []
        for doff, dsize in sorted(blocks):
            # Clipped to the start of the index.
            raw = self._read(doff, min(dsize, index_off - doff))
            if self.buf_size:
                raw = zlib.decompress(raw)
            rec = np.frombuffer(raw, dtype=dt)
            parts.append(rec[rec['chrom'] == cid])
        rec = np.concatenate(parts) if parts else np.zeros(0, dtype=dt)
        out = (
            rec['start'].astype(np.int64),
            rec['end'].astype(np.int64),
            rec['valid'].astype(np.float64),
            rec['sum'].astype(np.float64),
            rec['max'].astype(np.float64),
        )
        self._zoom_cache[key] = out
        return out

    def best_zoom(self, desired: int) -> int | None:
        """Kent v335 bbiBestZoom: closest reductionLevel <= desired; None when desired <= 1."""
        if desired <= 1:
            return None
        best, best_diff = None, None
        for i, (reduction, _) in enumerate(self.zooms):
            diff = desired - reduction
            if diff >= 0 and (best_diff is None or diff < best_diff):
                best, best_diff = i, diff
        return best

    def summary_mean(self, chrom: str, start0: int, end: int) -> float | None:
        """BinMean of bigWigSummaryArrayExtended(chrom, start0, end, 1); None if no data."""
        if start0 >= end:
            return None
        size = self.bw.chroms().get(chrom)
        if size is None:
            return None
        level = self.best_zoom((end - start0) // 2)
        if level is not None:
            starts, ends, valid, sums, _ = self._zoom_records(level, chrom)
            lo = np.searchsorted(ends, start0, side='right')
            hi = np.searchsorted(starts, end, side='left')
            count = 0.0
            total = 0.0
            for i in range(lo, hi):
                ov = min(end, ends[i]) - max(start0, starts[i])
                if ov > 0:
                    factor = ov / (ends[i] - starts[i])
                    count += valid[i] * factor
                    total += sums[i] * factor
            if count <= 0:
                return None
        else:
            ivs = self.bw.intervals(chrom, start0, end)
            if not ivs:
                return None
            count = 0.0
            total = 0.0
            for s, e, v in ivs:
                ov = min(end, e) - max(start0, s)
                if ov > 0:
                    size_i = e - s
                    weight = size_i * (ov / size_i)
                    count += weight
                    total += v * weight
        # normalizeCount: validCount = ceil(countFactor), sumData scaled to match; binMean divides back.
        valid_count = float(np.ceil(count))
        if valid_count == 0:
            return None
        return (total * (valid_count / count)) / valid_count

    def interval_gerp(self, chrom: str, a: int, b: int) -> float:
        """LOFTEE get_interval_gerp(chrom, a, b)."""
        size = self.bw.chroms().get(chrom)
        if size is None:
            return 0.0
        # get_seq_stream: `-start ||= 1; -end ||= size`, clip, then Bio::SeqFeature::Lite swaps.
        a = max(a or 1, 1)
        b = min(b or size, size)
        start, end = (a, b) if a <= b else (b, a)
        mean = self.summary_mean(chrom, start - 1, end)
        if mean is None:
            return 0.0
        return (end - start + 1) * mean

    def summary_max(self, chrom: str, start0: int, end: int) -> float | None:
        """MaxVal of bigWigSummaryArrayExtended(chrom, start0, end, 1); None if no data."""
        if start0 >= end or chrom not in self.chrom_ids:
            return None
        level = self.best_zoom((end - start0) // 2)
        if level is not None:
            starts, ends, valid, _, maxes = self._zoom_records(level, chrom)
            lo = np.searchsorted(ends, start0, side='right')
            hi = np.searchsorted(starts, end, side='left')
            hits = [maxes[i] for i in range(lo, hi) if min(end, ends[i]) - max(start0, starts[i]) > 0 and valid[i] > 0]
            return max(hits) if hits else None
        ivs = self.bw.intervals(chrom, start0, end)
        return max(v for _, _, v in ivs) if ivs else None


def gerp_span(start: int, end: int) -> tuple[int, int]:
    """The zero-based half-open bigWig window of a VEP span.

    An insertion (it ends before it starts) takes the two flanking bases.
    """
    return (start - 2, end + 1) if start - 1 == end else (start - 1, end)


def gerp_max(bigwig: str, chromosome: str, spans: pl.DataFrame) -> pl.Series:
    """The highest GERP score of each span, over the bigWig's full-resolution intervals.

    Args:
        bigwig: The GERP bigWig, a path pyBigWig can open.
        chromosome: The contig name as the bigWig spells it.
        spans: One row per span, with VEP's one-based `vepStart` and `vepEnd`.

    Returns:
        Float64, the numeric maximum, null where no base of the span is covered or the bigWig
        has no such contig.
    """
    starts, ends = spans['vepStart'].to_list(), spans['vepEnd'].to_list()
    values: dict[tuple[int, int], float | None] = {}
    with pyBigWig.open(bigwig) as handle:
        length = (handle.chroms() or {}).get(chromosome)
        if length is not None:
            for key in dict.fromkeys(zip(starts, ends, strict=True)):
                low, high = gerp_span(*key)
                low, high = max(low, 0), min(high, length)
                intervals = handle.intervals(chromosome, low, high) if low < high else None
                values[key] = max(value for _, _, value in intervals) if intervals else None
    return pl.Series('gerp', [values.get(key) for key in zip(starts, ends, strict=True)], dtype=pl.Float64)
