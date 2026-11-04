"""The mechanical check that every kept mechanism is backed by the trial text.

A quote counts only if it cites a trial of its pool, occurs in that trial's text (up to
whitespace) and names the target as the model wrote it.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pts.transformers.clinical_mira.moa.schemas import MoaExtraction

EVIDENCE = pl.List(pl.Struct({'nctId': pl.String, 'quote': pl.String}))

MECHANISM_SCHEMA = {
    'pool_id': pl.String,
    'mechanism_idx': pl.UInt32,
    'targetFromSource': pl.String,
    'targetSymbolSuggestion': pl.List(pl.String),
    'isHumanTarget': pl.Boolean,
    'targetType': pl.String,
    'actionType': pl.String,
    'mechanismOfAction': pl.String,
    'confidence': pl.Float64,
    'evidence': EVIDENCE,
    'kept': pl.Boolean,
    'drop_reason': pl.String,
}


def _ws(text: str) -> str:
    return re.sub(r'\s+', ' ', text).strip()


SHORT_SYMBOL = 6
"""Single-token targets up to this length are treated as gene symbols."""


def names_target(quote: str, target: str) -> bool:
    """Whether the quote contains the target as a whole word, ignoring case and whitespace.

    A short symbol written with capitals (``MET``, ``AR``, ``KIT``) does not match an
    all-lowercase word, which is an ordinary English word rather than the gene.
    """
    target = _ws(target)
    pattern = r'(?<![A-Za-z0-9])' + re.escape(target) + r'(?![A-Za-z0-9])'
    symbol = ' ' not in target and len(target) <= SHORT_SYMBOL and target != target.lower()
    for match in re.finditer(pattern, _ws(quote), flags=re.IGNORECASE):
        if not (symbol and match.group(0) == match.group(0).lower()):
            return True
    return False


def _quote_problem(nct_id: str, quote: str, target: str, pool_trials: set[str], texts: dict[str, str]) -> str | None:
    if not _ws(target):
        return 'empty_target'
    if nct_id not in pool_trials:
        return 'nct_not_in_pool'
    if _ws(quote) not in _ws(texts.get(nct_id, '')):
        return 'quote_not_in_text'
    if not names_target(quote, target):
        return 'target_not_in_quote'
    return None


def check_mechanisms(extractions: dict[str, MoaExtraction], pools: pl.DataFrame, texts: pl.DataFrame) -> pl.DataFrame:
    """Return one row per extracted mechanism, with only its valid quotes kept.

    Returns:
        pl.DataFrame: the columns of :py:data:`MECHANISM_SCHEMA`.
    """
    pool_trials = {
        row['pool_id']: set(row['nct_ids']) for row in pools.select('pool_id', 'nct_ids').iter_rows(named=True)
    }
    trial_text = dict(texts.select('id', 'prompt').iter_rows())

    rows = []
    for pool_id, extraction in extractions.items():
        trials = pool_trials.get(pool_id, set())
        for idx, mechanism in enumerate(extraction.mechanisms):
            problems = [
                _quote_problem(e.nct_id, e.quote, mechanism.target_from_source, trials, trial_text)
                for e in mechanism.evidence
            ]
            valid = [
                {'nctId': e.nct_id, 'quote': e.quote}
                for e, problem in zip(mechanism.evidence, problems, strict=True)
                if problem is None
            ]
            rows.append({
                'pool_id': pool_id,
                'mechanism_idx': idx,
                'targetFromSource': mechanism.target_from_source,
                'targetSymbolSuggestion': mechanism.target_symbol_suggestion,
                'isHumanTarget': mechanism.is_human_target,
                'targetType': mechanism.target_type.value,
                'actionType': mechanism.action_type.value,
                'mechanismOfAction': mechanism.mechanism_of_action,
                'confidence': mechanism.confidence,
                'evidence': valid,
                'kept': bool(valid),
                'drop_reason': None if valid else (problems[0] if problems else 'no_evidence'),
            })
    return pl.DataFrame(rows, schema=MECHANISM_SCHEMA)


def pool_status(pools: pl.DataFrame, extractions: dict[str, MoaExtraction], mechanisms: pl.DataFrame) -> pl.DataFrame:
    """Whether each pool was extracted and, if so, whether its texts state no mechanism.

    A pool states no mechanism when none of its mechanisms survived the quote check,
    whatever the model's own ``notStated`` flag says.

    Returns:
        pl.DataFrame: ``pool_id``, ``extracted``, ``not_stated`` (null when not extracted).
    """
    with_kept = set(mechanisms.filter(pl.col('kept'))['pool_id'].to_list())
    return (
        pools
        .select('pool_id')
        .with_columns(
            extracted=pl.col('pool_id').is_in(list(extractions)),
        )
        .with_columns(
            not_stated=pl.when(pl.col('extracted')).then(~pl.col('pool_id').is_in(list(with_kept))),
        )
    )
