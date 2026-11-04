"""Per-trial texts and the per-drug pools built from them.

A pool is every AACT trial whose extraction names exactly one investigated drug, grouped by
that drug's normalised name. Its text is the trials' rendered prompts, mechanism-bearing
trials first, cut at a character budget.
"""

from __future__ import annotations

import hashlib

import polars as pl

from pts.transformers.clinical_mira.aact_trial_extraction import prompts_for, trial_report
from pts.transformers.utils.aact_synonyms import EXACT_MATCH_BLOCKLIST, _normalize_name

MOA_VOCAB = (
    r'(?i)\b(?:inhibitors?|antagonists?|agonists?|blockers?|modulators?|targeting|targets|binds'
    r'|degraders?|activators?)\b|\banti-[a-z0-9]'
)
"""Words that suggest a trial text states a mechanism."""

TRIAL_SEPARATOR = '\n\n---\n\n'


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def trial_texts(archive: str, scratch_root: str) -> pl.DataFrame:
    """Rebuild every trial's prompt from the staged AACT archive with the staging code.

    Returns:
        pl.DataFrame: ``id``, ``prompt``, ``prompt_sha256``.
    """
    return prompts_for(trial_report(archive, scratch_root, None))


def match_texts(texts: pl.DataFrame, extraction: pl.DataFrame) -> tuple[pl.DataFrame, dict[str, int]]:
    """Keep the texts whose prompt hash equals the one the extraction was computed from.

    Returns:
        The matching texts, and counts of ``matched``, ``hash_mismatch`` and ``missing_text``
        over the extraction's trials.
    """
    joined = extraction.select('id', pl.col('prompt_sha256').alias('expected')).join(
        texts, on='id', how='left', maintain_order='left'
    )
    matched = joined.filter(pl.col('prompt_sha256') == pl.col('expected'))
    counts = {
        'matched': matched.height,
        'hash_mismatch': joined.filter(
            pl.col('expected').is_not_null()
            & pl.col('prompt_sha256').is_not_null()
            & (pl.col('prompt_sha256') != pl.col('expected'))
        ).height,
        'missing_text': joined.filter(pl.col('expected').is_not_null() & pl.col('prompt_sha256').is_null()).height,
        'missing_hash': joined.filter(pl.col('expected').is_null()).height,
    }
    return matched.select(texts.columns), counts


def single_drug_trials(extraction: pl.DataFrame) -> pl.DataFrame:
    """Trials whose extraction names exactly one investigated drug, with its normalised name.

    Returns:
        pl.DataFrame: ``nct_id``, ``drug_name``.
    """
    return (
        extraction
        .filter(pl.col('investigated_drugs').list.len() == 1)
        .select(
            pl.col('id').alias('nct_id'),
            _normalize_name(pl.col('investigated_drugs').list.first().struct.field('drug')).alias('normalised'),
        )
        .filter(pl.col('normalised').is_not_null() & ~pl.col('normalised').is_in(sorted(EXACT_MATCH_BLOCKLIST)))
        .with_columns(
            drug_name=pl
            .col('normalised')
            .str.replace_all(r'[^\w\s]', '')
            .str.replace_all(r'\s+', ' ')
            .str.strip_chars()
        )
        .filter(pl.col('drug_name').str.len_chars() > 0)
        .select('nct_id', 'drug_name')
    )


def build_pools(trials: pl.DataFrame, texts: pl.DataFrame, char_budget: int = 60_000) -> pl.DataFrame:
    """Pool the trials of each drug into one text.

    Returns:
        pl.DataFrame: ``pool_id``, ``drug_name``, ``nct_ids``, ``n_trials``, ``n_trials_used``,
        ``text``, ``pool_sha256``.
    """
    ordered = (
        trials
        .join(texts.select(pl.col('id').alias('nct_id'), 'prompt'), on='nct_id', how='inner')
        .with_columns(
            has_vocab=pl.col('prompt').str.contains(MOA_VOCAB),
            length=pl.col('prompt').str.len_chars(),
        )
        .sort(['drug_name', 'has_vocab', 'length', 'nct_id'], descending=[False, True, True, False])
        .with_columns(
            n_trials=pl.len().over('drug_name'),
            rank=pl.int_range(pl.len()).over('drug_name'),
            running=pl.col('length').cum_sum().over('drug_name'),
        )
    )
    used = ordered.filter((pl.col('rank') == 0) | (pl.col('running') <= char_budget))
    pools = used.group_by('drug_name', maintain_order=True).agg(
        nct_ids=pl.col('nct_id'),
        n_trials=pl.col('n_trials').first().cast(pl.UInt32),
        n_trials_used=pl.len().cast(pl.UInt32),
        text=pl.col('prompt').str.join(TRIAL_SEPARATOR),
    )
    return pools.with_columns(
        pool_id=pl.col('drug_name').map_elements(lambda n: _sha256(n)[:16], return_dtype=pl.String),
        pool_sha256=pl.col('text').map_elements(_sha256, return_dtype=pl.String),
    ).select('pool_id', 'drug_name', 'nct_ids', 'n_trials', 'n_trials_used', 'text', 'pool_sha256')
