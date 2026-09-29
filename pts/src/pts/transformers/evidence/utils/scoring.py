"""Evidence scoring.

Polars port of `calculate_evidence_score` in `pts.pyspark.evidence_utils.evidence.Evidence`. The
PySpark version evaluates an arbitrary SQL expression string per datasource; here a `pl.Expr` plays
that role instead -- a straight column copy (`pl.col('resourceScore')`, GWAS evidence) and a real
computation (a linear rescale, ENCORE evidence) are both just expressions.
"""

import polars as pl

from pts.transformers.evidence.utils import flags
from pts.transformers.utils.quality_flags import update_quality_flag


def calculate_evidence_score(df: pl.DataFrame, score_expr: pl.Expr) -> pl.DataFrame:
    """Add a `score` column from `score_expr`, flagging out-of-range or missing scores.

    Args:
        df: evidence dataframe.
        score_expr: expression computing the raw datasource-specific score, e.g. `pl.col(...)` for
            a straight copy, or a real computation such as a linear rescale.

    Returns:
        `df` with a new `score` column (`float64`), and `qualityControls` flagged with
        `flags.NO_VALID_SCORE` where `score` is null, negative or greater than 1.
    """
    scored = df.with_columns(score_expr.cast(pl.Float64).alias('score'))
    invalid = pl.col('score').is_null() | (pl.col('score') < 0) | (pl.col('score') > 1)
    return update_quality_flag(scored, invalid, flags.NO_VALID_SCORE)
