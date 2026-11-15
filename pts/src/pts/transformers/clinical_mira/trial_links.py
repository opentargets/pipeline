"""Turn v2 trial extractions into the drugs and diseases of a report, and into one row per link."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pts.transformers.clinical_mira.schemas import TrialExtractionV2

LINK_SCHEMA = {
    'id': pl.String,
    'drug': pl.String,
    'indication': pl.String,
    'broader': pl.String,
    'use': pl.String,
    'arms': pl.List(pl.String),
    'confidence': pl.Float64,
}


def report_entities(extraction: TrialExtractionV2) -> tuple[list[str], list[str]]:
    """The distinct drugs and indications of the links, sorted; background drugs and conditions are excluded."""
    drugs = sorted({link.drug for link in extraction.links})
    diseases = sorted({link.indication for link in extraction.links})
    return drugs, diseases


def link_rows(extractions: dict[str, TrialExtractionV2]) -> pl.DataFrame:
    """One row per link, keyed by the trial id the extractions are keyed by."""
    rows = [
        (trial_id, link.drug, link.indication, link.broader, link.use, link.arms, link.confidence)
        for trial_id, extraction in extractions.items()
        for link in extraction.links
    ]
    return pl.DataFrame(rows, schema=LINK_SCHEMA, orient='row')


def _links(extractions: pl.DataFrame) -> pl.DataFrame:
    """One row per link (``id`` plus the link fields); trials without links are absent."""
    return extractions.select('id', 'links').explode('links').drop_nulls('links').unnest('links')


def report_entities_frame(extractions: pl.DataFrame, min_confidence: float) -> pl.DataFrame:
    """Per trial, the distinct sorted drugs and indications of links at or above ``min_confidence``.

    Trials with no such link keep a row with empty lists. ``diseaseBroader`` holds one ``{indication, broader}``
    struct per distinct indication, with the first broader seen.
    """
    kept = _links(extractions).filter(pl.col('confidence') >= min_confidence)
    ids = extractions.select('id')
    grouped = kept.group_by('id').agg(
        pl.col('drug').unique().sort().alias('drugFromSource'),
        pl.col('indication').unique().sort().alias('diseaseFromSource'),
    )
    broader = (
        kept
        .unique(subset=['id', 'indication'], keep='first', maintain_order=True)
        .group_by('id', maintain_order=True)
        .agg(pl.struct('indication', 'broader').alias('diseaseBroader'))
    )
    empty = pl.lit([], dtype=pl.List(pl.String))
    return (
        ids
        .join(grouped, on='id', how='left', maintain_order='left')
        .join(broader, on='id', how='left', maintain_order='left')
        .with_columns(
            pl.col('drugFromSource').fill_null(empty),
            pl.col('diseaseFromSource').fill_null(empty),
            pl.col('diseaseBroader').fill_null(
                pl.lit([], dtype=pl.List(pl.Struct({'indication': pl.String, 'broader': pl.String})))
            ),
        )
        .rename({'id': 'nct_id'})
    )


def tested_drug_labels(extractions: pl.DataFrame) -> pl.DataFrame:
    """The distinct, non-null link drugs per trial, whatever their confidence."""
    return (
        _links(extractions).select('id', pl.col('drug').alias('label')).drop_nulls('label').unique(maintain_order=True)
    )


def drug_synonym_entries(extractions: pl.DataFrame) -> pl.DataFrame:
    """One row per drug entry: link drugs as role ``tested`` (per trial and drug, synonyms unioned), then role drugs."""
    tested = (
        _links(extractions)
        .group_by('id', 'drug', maintain_order=True)
        .agg(pl.col('synonyms').explode().drop_nulls().unique(maintain_order=True))
        .with_columns(pl.lit('tested').alias('role'))
    )
    roles = [
        extractions
        .select('id', column)
        .explode(column)
        .drop_nulls(column)
        .unnest(column)
        .select('id', 'drug', 'synonyms', 'role')
        for column in ('background_drugs', 'other_drugs')
    ]
    return pl.concat([tested.select('id', 'drug', 'synonyms', 'role'), *roles])


def single_drug_trials(extractions: pl.DataFrame) -> pl.DataFrame:
    """Trials whose links name exactly one distinct drug (case-insensitive, trimmed); the drug as first written."""
    return (
        _links(extractions)
        .with_columns(pl.col('drug').str.strip_chars().str.to_lowercase().alias('_key'))
        .group_by('id', maintain_order=True)
        .agg(pl.col('_key').n_unique().alias('_n'), pl.col('drug').first())
        .filter(pl.col('_n') == 1)
        .select('id', 'drug')
    )
