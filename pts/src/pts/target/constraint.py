"""gnomAD genetic constraint per gene."""

from __future__ import annotations

import polars as pl

from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_casts import to_float, to_int

METRICS = {
    'syn': ['syn.z_score', 'syn.exp', 'syn.obs', 'syn.oe', 'syn.oe_ci.lower', 'syn.oe_ci.upper'],
    'mis': ['mis.z_score', 'mis.exp', 'mis.obs', 'mis.oe', 'mis.oe_ci.lower', 'mis.oe_ci.upper'],
    'lof': ['lof.pLI', 'lof.exp', 'lof.obs', 'lof.oe', 'lof.oe_ci.lower', 'lof.oe_ci.upper'],
}
RANK = 'lof.oe_ci.upper_rank'
COLUMNS = ['gene_id', 'canonical', 'transcript_type', RANK, *[c for cols in METRICS.values() for c in cols]]


def read_constraint(path: str) -> pl.DataFrame:
    """The gnomAD constraint metrics TSV, the columns used here, as text."""
    return scan_dataset(path, format='tsv', has_header=True, infer_schema=False).select(COLUMNS).collect()


def ntile(n: int, count: pl.Expr, position: pl.Expr) -> pl.Expr:
    """Spark's `ntile(n) - 1` for the row at 0-based `position` of `count` rows."""
    size = count // n
    remainder = count % n
    large = remainder * (size + 1)
    return (
        pl
        .when(position < large)
        .then(position // (size + 1))
        .otherwise(remainder + (position - large) // pl.max_horizontal(size, 1))
        .cast(pl.Int32)
    )


def _constraint(kind: str, upper_rank: pl.Expr, upper_bin: pl.Expr, upper_bin6: pl.Expr) -> pl.Expr:
    score, exp, obs, oe, lower, upper = METRICS[kind]
    return pl.struct(
        pl.lit(kind).alias('constraintType'),
        to_float(pl.col(score)).alias('score'),
        to_float(pl.col(exp)).alias('exp'),
        to_int(pl.col(obs)).alias('obs'),
        to_float(pl.col(oe)).alias('oe'),
        to_float(pl.col(lower)).alias('oeLower'),
        to_float(pl.col(upper)).alias('oeUpper'),
        upper_rank.alias('upperRank'),
        upper_bin.alias('upperBin'),
        upper_bin6.alias('upperBin6'),
    )


def build_constraint(raw: pl.DataFrame) -> pl.DataFrame:
    """syn, mis and lof constraint for each canonical transcript's gene.

    The lof decile and sextile bins are computed over the genes that carry a LOEUF rank.

    Returns:
        [id, constraint].
    """
    canonical = raw.filter((pl.col('canonical') == 'true') & (pl.col('transcript_type') != 'NA'))
    ranked = canonical.filter(pl.col(RANK) != 'NA').sort(to_int(pl.col(RANK)), nulls_last=False, maintain_order=True)
    position = pl.int_range(pl.len(), dtype=pl.Int64)
    count = pl.len().cast(pl.Int64)
    bins = ranked.select(
        'gene_id',
        ntile(10, count, position).alias('lof_upper_bin'),
        ntile(6, count, position).alias('lof_upper_bin6'),
    )
    null_int = pl.lit(None, dtype=pl.Int32)
    return canonical.join(bins, on='gene_id', how='left', maintain_order='left').select(
        pl.col('gene_id').alias('id'),
        pl.concat_list(
            _constraint('syn', null_int, null_int, null_int),
            _constraint('mis', null_int, null_int, null_int),
            _constraint('lof', to_int(pl.col(RANK)), pl.col('lof_upper_bin'), pl.col('lof_upper_bin6')),
        ).alias('constraint'),
    )
