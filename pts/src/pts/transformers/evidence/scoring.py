"""Evidence scoring.

Polars port of `calculate_evidence_score` in `pts.pyspark.evidence_utils.evidence.Evidence`. The
PySpark version evaluates an arbitrary SQL expression per datasource; every Polars-migrated
datasource so far only ever needs a straight column copy, so this takes a column name rather than
an expression. Extend it to accept a `pl.Expr` if a future datasource needs a real computation.
"""

import polars as pl

from pts.transformers.evidence import flags
from pts.transformers.utils.quality_flags import update_quality_flag


def calculate_evidence_score(df: pl.DataFrame, score_column: str) -> pl.DataFrame:
    """Add a `score` column from `score_column`, flagging out-of-range or missing scores.

    Args:
        df: evidence dataframe.
        score_column: name of the column holding the raw datasource-specific score.

    Returns:
        `df` with a new `score` column (`float64`), and `qualityControls` flagged with
        `flags.NO_VALID_SCORE` where `score` is null, negative or greater than 1.
    """
    scored = df.with_columns(pl.col(score_column).cast(pl.Float64).alias('score'))
    invalid = pl.col('score').is_null() | (pl.col('score') < 0) | (pl.col('score') > 1)
    return update_quality_flag(scored, invalid, flags.NO_VALID_SCORE)
