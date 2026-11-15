"""The AACT trial report and the per-trial prompts rendered from it.

:py:mod:`pts.transformers.clinical_mira.aact_trial_extraction_v2` takes its trials from
:py:func:`trial_report`; the MoA staging pools registry text from :py:func:`prompts_for`.
"""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

import polars as pl
from loguru import logger

from pts.transformers.clinical_mira import build_prompts, extract_aact_clinical_report, sample_report
from pts.transformers.clinical_mira.providers.aact import read_aact_tables
from pts.transformers.clinical_mira.providers.aact_llm import TRIAL_FIELDS
from pts.transformers.clinical_mira.record_store import text_sha256

if TYPE_CHECKING:
    from pathlib import Path

IMPORT_COMPUTED_AT = datetime(1970, 1, 1, tzinfo=UTC)
"""The time of imported records cached without one."""


def trial_report(archive: str, scratch_root: str | Path, sample_size: int | None) -> pl.DataFrame:
    """Join the AACT tables into one row per trial.

    This is only the AACT side of the picture. The release dataset of the
    same name unions several other sources and maps entities against the
    disease and molecule indexes, none of which is needed — or available —
    here.
    """
    logger.info(f'restoring AACT tables from {archive}')
    tables = read_aact_tables(
        archive,
        ['studies', 'interventions', 'conditions', 'study_references', 'brief_summaries', 'detailed_descriptions'],
        scratch_root,
    )

    report = extract_aact_clinical_report(
        studies=tables['studies'].select('nct_id', 'study_type', 'phase', 'official_title'),
        interventions=tables['interventions'],
        conditions=tables['conditions'],
        additional_metadata=[
            tables['study_references'],
            tables['brief_summaries'],
            tables['detailed_descriptions'].rename({'description': 'detailed_description'}),
        ],
        aggregation_specs={'pmid': {'group_by': 'nct_id', 'alias': 'literature'}},
    )
    return sample_report(report.df, sample_size)


def prompts_for(report: pl.DataFrame) -> pl.DataFrame:
    """Render one prompt per trial, with the hash of the prompt.

    The prompt is the record store's text. The response schema does not enter
    it; the store keys the schema digest as the model revision instead.
    """
    if 'drugs' in report.columns:
        report = report.with_columns(
            pl.col('drugs').list.eval(pl.element().sort_by(pl.element().struct.field('drugFromSource')))
        )
    prompts = build_prompts(report, trial_fields=TRIAL_FIELDS)

    return pl.DataFrame(prompts, schema={'id': pl.String, 'prompt': pl.String}).with_columns(
        prompt_sha256=pl.col('prompt').map_elements(text_sha256, return_dtype=pl.String),
    )
