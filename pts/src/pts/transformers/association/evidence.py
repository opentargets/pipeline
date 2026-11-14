"""Evidence preparation for association scoring."""

from __future__ import annotations

import polars as pl


def evidence_year(evidence_date: pl.Expr, current_year: int) -> pl.Expr:
    """Leading four-digit year capped at `current_year`; `current_year + 1` when there is none."""
    year = evidence_date.str.extract(r'^([0-9]{4})(?:[^0-9]|$)', 1).str.strip_chars(' ').cast(pl.Int32, strict=False)
    return year.clip(upper_bound=current_year).fill_null(pl.lit(current_year + 1, dtype=pl.Int32))


def prepare_evidence(raw: pl.LazyFrame, current_year: int) -> pl.LazyFrame:
    """Keep positive-score evidence with its year."""
    return raw.filter(pl.col('score') > 0).select(
        'targetId',
        'diseaseId',
        'datasourceId',
        'datatypeId',
        pl.col('score').cast(pl.Float64),
        evidence_year(pl.col('evidenceDate'), current_year).alias('year'),
    )


def expand_disease(evidence: pl.LazyFrame, disease: pl.LazyFrame, weights: pl.DataFrame) -> pl.LazyFrame:
    """Propagate evidence to each disease's ancestors; non-propagated datasources stay direct."""
    columns = evidence.collect_schema().names()
    flagged = evidence.join(
        weights.select('datasourceId', 'toPropagate').lazy(), on='datasourceId', how='left'
    ).with_columns(pl.col('toPropagate').fill_null(True))
    propagated = flagged.filter(pl.col('toPropagate')).drop('toPropagate')
    direct = flagged.filter(~pl.col('toPropagate')).drop('toPropagate')

    lineage = (
        disease
        .select(
            pl.col('id').alias('diseaseId'),
            pl.concat_list(pl.col('id'), pl.col('ancestors')).list.unique().alias('specificDiseaseId'),
        )
        .explode('specificDiseaseId')
        .drop_nulls('specificDiseaseId')
    )
    expanded = (
        propagated
        .join(lineage, on='diseaseId', how='inner')
        .drop('diseaseId')
        .rename({'specificDiseaseId': 'diseaseId'})
    )
    return pl.concat([expanded.select(columns), direct.select(columns)])
