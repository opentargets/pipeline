"""Quality flag utilities for datasets processed with Polars."""

import polars as pl


def update_quality_flag(
    df: pl.DataFrame,
    flag_condition: pl.Expr,
    flag_text: str,
) -> pl.DataFrame:
    """Update the qualityControls column in a Polars DataFrame based on a flag condition.

    Args:
        df: The input Polars DataFrame
        flag_condition: A Polars expression that evaluates to a boolean
        flag_text: The text to add to the qualityControls list if the condition is met

    Returns:
        A new Polars DataFrame with updated qualityControls column
    """
    # First ensure qualityControls column exists
    if 'qualityControls' not in df.columns:
        df = df.with_columns(pl.lit(None, dtype=pl.List(pl.String)).alias('qualityControls'))

    return df.with_columns(
        pl.when(flag_condition)
        .then(
            pl.concat_list([
                pl.col('qualityControls').fill_null([]),
                pl.lit(flag_text).implode(),
            ])
            .list.unique()
            .list.sort()
        )
        .otherwise(pl.col('qualityControls'))
        .alias('qualityControls')
    )
