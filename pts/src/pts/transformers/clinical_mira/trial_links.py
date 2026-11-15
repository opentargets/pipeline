"""Turn v2 trial extractions into the drugs and diseases of a report, and into one row per link."""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pts.transformers.clinical_mira.schemas import TrialExtractionV2

LINK_SCHEMA = {'id': pl.String, 'drug': pl.String, 'indication': pl.String, 'use': pl.String, 'confidence': pl.Float64}


def report_entities(extraction: TrialExtractionV2) -> tuple[list[str], list[str]]:
    """The distinct drugs and indications of the links, sorted; background drugs and conditions are excluded."""
    drugs = sorted({link.drug for link in extraction.links})
    diseases = sorted({link.indication for link in extraction.links})
    return drugs, diseases


def link_rows(extractions: dict[str, TrialExtractionV2]) -> pl.DataFrame:
    """One row per link, keyed by the trial id the extractions are keyed by."""
    rows = [
        (trial_id, link.drug, link.indication, link.use, link.confidence)
        for trial_id, extraction in extractions.items()
        for link in extraction.links
    ]
    return pl.DataFrame(rows, schema=LINK_SCHEMA, orient='row')
