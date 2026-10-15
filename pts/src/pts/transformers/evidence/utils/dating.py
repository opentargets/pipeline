"""Evidence dating: resolving publication and overall evidence dates.

Polars port of `resolve_publication_date`/`resolve_evidence_date` in
`pts.pyspark.evidence_utils.evidence.Evidence`.
"""

import polars as pl

#: Columns considered when resolving the overall `evidenceDate`, in the same order as the
#: PySpark `Evidence.EVIDENCE_DATE_COLUMNS` default.
EVIDENCE_DATE_COLUMNS = ('publicationDate', 'curationDate', 'studyStartDate', 'releaseDate', 'evidenceDate')


def resolve_publication_date(df: pl.DataFrame, publication_lut: pl.DataFrame) -> pl.DataFrame:
    """Add a `publicationDate` column from the earliest dated reference in `literature`.

    A no-op (returns `df` unchanged) when `df` has no `literature` column, matching the PySpark
    behaviour -- `literature` is optional, `id` is not.

    Args:
        df: evidence dataframe with `id` and (optionally) `literature` (list[str]) columns.
        publication_lut: output of `luts.build_publication_lut`.

    Returns:
        `df` with a new `publicationDate` column, null where no reference in `literature` matched
        the look-up table.
    """
    if 'literature' not in df.columns:
        return df

    evidence_with_pub_ids = (
        df
        .select('id', 'literature')
        .explode('literature', empty_as_null=True)
        .rename({'literature': 'publicationId'})
        .drop_nulls('publicationId')
        .with_columns(pl.col('publicationId').str.to_uppercase().str.strip_chars())
        .unique()
    )

    dated_evidence = (
        evidence_with_pub_ids
        .join(publication_lut, on='publicationId', how='inner')
        .sort('publicationDate')
        .group_by('id', maintain_order=True)
        .agg(pl.col('publicationDate').first())
    )

    return df.join(dated_evidence, on='id', how='left')


def resolve_evidence_date(
    df: pl.DataFrame,
    date_columns: tuple[str, ...] = EVIDENCE_DATE_COLUMNS,
) -> pl.DataFrame:
    """Add an `evidenceDate` column: the earliest non-null value across `date_columns`.

    `evidenceDate` is always present in the output, even when none of `date_columns` are, in
    which case it is entirely null.

    Args:
        df: evidence dataframe.
        date_columns: candidate date columns, in ISO `YYYY-MM-DD`-comparable string form.

    Returns:
        `df` with a new `evidenceDate` column.
    """
    present_columns = [column for column in date_columns if column in df.columns]
    if not present_columns:
        return df.with_columns(pl.lit(None, dtype=pl.String).alias('evidenceDate'))

    return df.with_columns(
        pl
        .concat_list([pl.col(column) for column in present_columns])
        .list.drop_nulls()
        .list.min()
        .alias('evidenceDate')
    )
