"""Per-trial texts and the per-drug pools built from them.

A pool is every AACT trial whose extraction names exactly one investigated drug, grouped by
that drug's normalised name. Its text is the trials' rendered prompts, mechanism-bearing
trials first, cut at a character budget.
"""

from __future__ import annotations

import hashlib
import re

import polars as pl

from pts.transformers.clinical_mira.moa.papers import paper_id, paper_text
from pts.transformers.utils.aact_synonyms import EXACT_MATCH_BLOCKLIST, _normalize_name

MOA_VOCAB = (
    r'(?i)\b(?:inhibitors?|antagonists?|agonists?|blockers?|modulators?|targeting|targets|binds'
    r'|degraders?|activators?)\b|\banti-[a-z0-9]'
)
"""Words that suggest a trial text states a mechanism."""


def mentions_mechanism(text: pl.Expr) -> pl.Expr:
    """Whether a text matches :py:data:`MOA_VOCAB`, case-insensitively."""
    return text.str.contains(MOA_VOCAB)


TRIAL_SEPARATOR = '\n\n---\n\n'

CLASS_NAME = (
    r'(?:inhibitors|agonists|antagonists|antibodies|analogues|analogs|statins|steroids|corticosteroids|nsaids'
    r'|opioids|benzodiazepines|diuretics|analgesics|anticoagulants|antibiotics|antivirals|antifungals'
    r'|antiplatelets|antihypertensives|antidepressants|antipsychotics|antiepileptics|anticonvulsants'
    r'|bisphosphonates|blockers|vaccines|drugs|agents|medications|medicines|supplements|probiotics|vitamins'
    r'|anesthetics|anaesthetics|sedatives|taxanes|anthracyclines|fluoroquinolones|macrolides|sulfonylureas'
    r'|cephalosporins|penicillins|carbapenems|triptans|gliptins|glucocorticoids|immunosuppressants'
    r'|chemotherapy|immunotherapy|radiotherapy|hormone therapy|standard of care|placebo)$'
)
"""A name ending in a plural drug-class noun, or a treatment category, names a class, not a drug."""


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


CONTENT_FIELDS = ('Official Title:', 'Description:', 'Detailed Description:')
"""Prompt fields a trial's content key covers; the rest (interventions, conditions, phase) is left out."""


def trial_content_sha256(prompt: str) -> str:
    """Hash a trial's title and descriptions, whitespace-normalised, ignoring how the prompt is rendered."""
    lines, keep = [], False
    for line in prompt.splitlines():
        if line.startswith(CONTENT_FIELDS):
            keep = True
        elif re.match(r'^[A-Z][A-Za-z ]+:', line):
            keep = False
        if keep:
            lines.append(line)
    return _sha256(re.sub(r'\s+', ' ', '\n'.join(lines)).strip())


def content_key(drug_name: str, entries: list[tuple[str, str]]) -> str:
    """Hash a drug name and its ``(id, content sha256)`` entries, whatever their order."""
    return _sha256(drug_name + '\x1f' + '|'.join(f'{entry_id}:{sha}' for entry_id, sha in sorted(entries)))


def trial_texts(archive: str, scratch_root: str) -> pl.DataFrame:
    """Rebuild every trial's prompt from the staged AACT archive with the staging code.

    Returns:
        pl.DataFrame: ``id``, ``prompt``, ``prompt_sha256``.
    """
    from pts.transformers.clinical_mira.aact_trial_extraction import prompts_for, trial_report

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
        .filter((pl.col('drug_name').str.len_chars() > 0) & ~pl.col('drug_name').str.contains(CLASS_NAME))
        .select('nct_id', 'drug_name')
    )


def build_pools(trials: pl.DataFrame, texts: pl.DataFrame, char_budget: int = 60_000) -> pl.DataFrame:
    """Pool the trials of each drug into one text.

    Returns:
        pl.DataFrame: ``pool_id``, ``drug_name``, ``nct_ids``, ``n_trials``, ``n_trials_used``,
        ``text``, ``pool_sha256``, ``content_key``. ``content_key`` hashes the drug name and
        every trial's id and content (not only the trials within the budget), so it changes
        exactly when a trial is added, removed or edited.
    """
    ordered = (
        trials
        .join(texts.select(pl.col('id').alias('nct_id'), 'prompt'), on='nct_id', how='inner')
        .with_columns(
            has_vocab=mentions_mechanism(pl.col('prompt')),
            length=pl.col('prompt').str.len_chars(),
            content=pl.col('prompt').map_elements(trial_content_sha256, return_dtype=pl.String),
        )
        .sort(['drug_name', 'has_vocab', 'length', 'nct_id'], descending=[False, True, True, False])
        .with_columns(
            n_trials=pl.len().over('drug_name'),
            rank=pl.int_range(pl.len()).over('drug_name'),
            running=pl.col('length').cum_sum().over('drug_name'),
        )
    )
    keys = (
        ordered
        .group_by('drug_name', maintain_order=True)
        .agg(entries=pl.struct('nct_id', 'content'))
        .with_columns(
            content_key=pl.struct('drug_name', 'entries').map_elements(
                lambda row: content_key(row['drug_name'], [(e['nct_id'], e['content']) for e in row['entries']]),
                return_dtype=pl.String,
            )
        )
        .select('drug_name', 'content_key')
    )
    used = ordered.filter((pl.col('rank') == 0) | (pl.col('running') <= char_budget))
    pools = used.group_by('drug_name', maintain_order=True).agg(
        nct_ids=pl.col('nct_id'),
        n_trials=pl.col('n_trials').first().cast(pl.UInt32),
        n_trials_used=pl.len().cast(pl.UInt32),
        text=pl.col('prompt').str.join(TRIAL_SEPARATOR),
    )
    return (
        pools
        .join(keys, on='drug_name', how='left')
        .with_columns(
            pool_id=pl.col('drug_name').map_elements(lambda n: _sha256(n)[:16], return_dtype=pl.String),
            pool_sha256=pl.col('text').map_elements(_sha256, return_dtype=pl.String),
        )
        .select('pool_id', 'drug_name', 'nct_ids', 'n_trials', 'n_trials_used', 'text', 'pool_sha256', 'content_key')
    )


def add_papers(
    pools: pl.DataFrame,
    trials: pl.DataFrame,
    texts: pl.DataFrame,
    selected: pl.DataFrame,
    abstracts: pl.DataFrame,
    abstract_chars: int,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Append each pool's publications that have an abstract to its text, ids and content key.

    Publications are appended newest first. The content key is recomputed over every trial of the
    drug and the pool's publications; pools without publications are returned unchanged.

    Returns:
        ``(pools, trials, texts)``, the last two with a row per added publication.
    """
    rendered = (
        selected
        .join(abstracts.select('pmid', 'title', 'abstract'), on='pmid', how='inner', maintain_order='left')
        .filter(pl.col('abstract').is_not_null() & (pl.col('abstract').str.strip_chars().str.len_chars() > 0))
        .join(pools.select('drug_name'), on='drug_name', how='semi', maintain_order='left')
        .sort(pl.col('pmid').cast(pl.Int64), descending=True)
        .with_columns(
            id=pl.col('pmid').map_elements(paper_id, return_dtype=pl.String),
            prompt=pl.struct('pmid', 'title', 'abstract').map_elements(
                lambda r: paper_text(r['pmid'], r['title'], r['abstract'], abstract_chars), return_dtype=pl.String
            ),
        )
        .with_columns(prompt_sha256=pl.col('prompt').map_elements(_sha256, return_dtype=pl.String))
    )
    if rendered.is_empty():
        return pools, trials, texts

    trial_entries = (
        trials
        .select('nct_id', 'drug_name')
        .join(texts.select(pl.col('id').alias('nct_id'), 'prompt'), on='nct_id', how='inner', maintain_order='left')
        .with_columns(content=pl.col('prompt').map_elements(trial_content_sha256, return_dtype=pl.String))
    )
    entries: dict[str, list[tuple[str, str]]] = {}
    for row in trial_entries.iter_rows(named=True):
        entries.setdefault(row['drug_name'], []).append((row['nct_id'], row['content']))
    appended: dict[str, list[dict[str, str]]] = {}
    for row in rendered.iter_rows(named=True):
        appended.setdefault(row['drug_name'], []).append(row)

    rows = []
    for pool in pools.iter_rows(named=True):
        added = appended.get(pool['drug_name'])
        if added:
            drug_entries = entries.get(pool['drug_name'], []) + [
                (a['id'], trial_content_sha256(a['prompt'])) for a in added
            ]
            text = TRIAL_SEPARATOR.join([pool['text'], *(a['prompt'] for a in added)])
            pool = {
                **pool,
                'nct_ids': [*pool['nct_ids'], *(a['id'] for a in added)],
                'text': text,
                'pool_sha256': _sha256(text),
                'content_key': content_key(pool['drug_name'], drug_entries),
            }
        rows.append(pool)
    new_pools = pl.DataFrame(rows, schema=pools.schema)

    new_trials = rendered.select(pl.col('id').alias('nct_id'), 'drug_name')
    new_texts = (
        rendered
        .select('id', 'prompt', 'prompt_sha256')
        .unique('id', keep='first', maintain_order=True)
        .join(texts.select('id'), on='id', how='anti', maintain_order='left')
    )
    return (
        new_pools,
        pl.concat([trials, new_trials.select(trials.columns)], how='vertical_relaxed'),
        pl.concat([texts, new_texts.select(texts.columns)], how='vertical_relaxed'),
    )
