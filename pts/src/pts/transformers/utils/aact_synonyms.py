"""AACT clinical-trial synonym mining for ChEMBL molecules.

Mines candidate drug synonyms from the OpenAI/AACT clinical-trial extraction and anchors
them to ChEMBL molecules. :mod:`pts.transformers.chembl_molecule` is the only consumer and
reaches it through three functions: :func:`parse_aact_entries`,
:func:`mine_aact_synonyms` and :func:`merge_aact_synonyms`.

Pipeline: parsed batch output (from ``clinical_mining``'s ``parse_batch_results``) ->
normalized drug member sets -> anchor against a ChEMBL name index (with an ambiguity cap)
-> eleven cleanup rules -> keep candidates seen in ``MIN_TRIALS`` distinct trials -> merge
into the molecule synonyms.
"""

import polars as pl
from clinical_mining.schemas import ClinicalSource

AACT_SOURCE = ClinicalSource.AACT.value

# --- tunables ----------------------------------------------------------------

AMBIGUITY_CAP = 10
MIN_TRIALS = 2

DRUG_CODE_REGEX = r'\b[a-z]{1,6}-?\d{3,}[a-z0-9]*\b'

CONTROL_TERMS = {
    'placebo',
    'vehicle',
    'saline',
    'sham',
    'soc',
    'standard of care',
    'study drug',
    'sodium chloride',
    'water',
    'air',
    'normal saline',
}
CLASS_KEYWORDS = [
    'inhibitor',
    'agonist',
    'antagonist',
    'antibody',
    'analogue',
    'analog',
    'therapy',
    'statin',
    'steroid',
    'nsaid',
    'cell',
    'cells',
    'lymphocyte',
    'lymphocytes',
    'mesenchymal',
    'stromal',
    'progenitor',
    'fibroblast',
]
_CLASS_PATTERN = r'\b(' + '|'.join(CLASS_KEYWORDS) + r')\b'


def _normalize_name(expr: pl.Expr) -> pl.Expr:
    r"""Lowercase, strip trademark symbols, trim, collapse internal whitespace.

    The ``\s+`` collapse is Unicode-aware, so a non-breaking space folds like a plain
    one -- two arms spelling the same dose differently must not become two candidates.
    """
    stripped = expr.str.replace_all(r'[®™©℠]', '')
    collapsed = stripped.str.strip_chars().str.replace_all(r'\s+', ' ')
    return collapsed.str.to_lowercase()


def _has_class_keyword(expr: pl.Expr) -> pl.Expr:
    """True when the candidate text contains any drug-class / cell-therapy keyword as a whole word."""
    return expr.str.contains(_CLASS_PATTERN)


def parse_aact_entries(batch: pl.DataFrame) -> pl.DataFrame:
    """Turn parsed AACT batch extractions into one row per drug entry with a normalized member set.

    Args:
        batch: DataFrame as returned by
            :func:`clinical_mining.data_sources.aact.llm_extractor.parse_batch_results`
            (or, for tests, anything carrying the same ``id``,
            ``investigated_drugs``/``comparator_drugs``/``supportive_drugs`` columns).

    Returns:
        DataFrame[nct_id, members: list[str]] (normalized, deduped, non-empty).
    """
    roles = pl.concat_list(
        pl.col('investigated_drugs').fill_null([]),
        pl.col('comparator_drugs').fill_null([]),
        pl.col('supportive_drugs').fill_null([]),
    )
    return (
        batch
        .select(pl.col('id').alias('nct_id'), roles.alias('entry'))
        .explode('entry')
        # a trial with no extracted drugs explodes to a null row rather than to nothing
        .drop_nulls('entry')
        .unnest('entry')
        .with_columns(members=pl.concat_list(pl.concat_list(pl.col('drug')), pl.col('synonyms').fill_null([])))
        .with_columns(
            members=pl
            .col('members')
            .list.eval(_normalize_name(pl.element()))
            .list.eval(pl.element().filter(pl.element().is_not_null() & (pl.element().str.len_chars() > 0)))
            .list.unique(maintain_order=True)
        )
        .filter(pl.col('members').list.len() > 0)
        .select('nct_id', 'members')
    )


def _build_chembl_indexes(mol_df: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build (name_index, regimen_index, parent_child) from ChEMBL-source names.

    Args:
        mol_df: Molecule DataFrame with non-null id, name, synonyms, tradeNames,
            parentId, childChemblIds columns (synonyms/tradeNames/childChemblIds
            already filled to ``[]`` where absent).

    Returns:
        name_index:    DataFrame[name_norm, ids: list[str]]
        regimen_index: DataFrame[regimen_norm, ids: list[str]]  (suppression only)
        parent_child:  DataFrame[id, related: list[str]]  (parent + children)
    """
    labels = (
        mol_df
        .select(
            'id',
            pl.concat_list(
                pl.concat_list(pl.col('name')),
                pl.concat_list(
                    pl.col('synonyms').list.eval(pl.element().struct.field('label')),
                    pl.col('tradeNames').list.eval(pl.element().struct.field('label')),
                ),
            ).alias('labels'),
        )
        .explode('labels')
        .rename({'labels': 'label'})
        .with_columns(name_norm=_normalize_name(pl.col('label')))
        .filter(pl.col('name_norm').str.len_chars() > 0)
    )

    name_index = labels.group_by('name_norm').agg(pl.col('id').drop_nulls().unique().alias('ids'))

    # "<ingredient> COMPONENT OF <regimen>" -> regimen token (normalized text is lowercased)
    regimen_index = (
        labels
        .with_columns(regimen_norm=pl.col('name_norm').str.extract(r'\bcomponent of\s+(.+)$', 1))
        .filter(pl.col('regimen_norm').is_not_null() & (pl.col('regimen_norm').str.len_chars() > 0))
        .group_by('regimen_norm')
        .agg(pl.col('id').drop_nulls().unique().alias('ids'))
    )

    children = mol_df.select('id', pl.col('childChemblIds').alias('related'))
    parents = mol_df.filter(pl.col('parentId').is_not_null()).select(
        'id', pl.concat_list(pl.col('parentId')).alias('related')
    )
    parent_child = (
        pl
        .concat([children, parents])
        .group_by('id')
        .agg(
            pl
            .col('related')
            .list.explode(keep_nulls=False, empty_as_null=False)
            .unique(maintain_order=True)
            .alias('related')
        )
    )

    return name_index, regimen_index, parent_child


def _anchor_candidates(entries: pl.DataFrame, name_index: pl.DataFrame, parent_child: pl.DataFrame) -> pl.DataFrame:
    """Anchor member sets to molecules and emit (id, candidate, nct_id, status).

    For each trial drug entry (a normalized member set), resolve members against
    name_index to find which ChEMBL molecule(s) the entry anchors to, then emit each
    member that is NOT already on an anchored molecule as a candidate synonym,
    classified by status.

    Entries where any single member resolves to more than AMBIGUITY_CAP molecules are
    dropped entirely.

    Note: the same (id, candidate, nct_id) may appear with more than one status when a
    trial contributes multiple drug entries; downstream trial counting must use
    n_unique(nct_id).

    Args:
        entries: DataFrame[nct_id, members: list[str]]
        name_index: DataFrame[name_norm, ids: list[str]]
        parent_child: DataFrame[id, related: list[str]]

    Returns:
        DataFrame[id, candidate, nct_id, status] where id is an anchored molecule,
        candidate is a member not already on id, and status is one of
        NOVEL / PARENT_CHILD / CONFLICT.
    """
    # Deterministic per-entry key (nct_id + sorted member set), joined on a control
    # character that cannot occur in a normalized name or an NCT id, so two distinct
    # entries cannot collide onto the same key. A plain string rather than a hash:
    # collisions here would silently merge unrelated trials.
    entries = entries.with_columns(
        entry_id=pl.col('nct_id') + pl.lit('\x1f') + pl.col('members').list.sort().list.join('\x1f')
    )

    members = entries.select('entry_id', 'nct_id', pl.col('members').alias('member')).explode('member')

    resolved = (
        members
        .join(name_index, left_on='member', right_on='name_norm', how='left')
        .with_columns(ids=pl.col('ids').fill_null([]))
        .select('entry_id', 'nct_id', 'member', 'ids')
    )

    poisoned = (
        resolved
        .group_by('entry_id')
        .agg(pl.col('ids').list.len().max().alias('max_ids'))
        .filter(pl.col('max_ids') > AMBIGUITY_CAP)
        .select('entry_id')
    )
    resolved = resolved.join(poisoned, on='entry_id', how='anti')

    anchors = (
        resolved
        .select('entry_id', pl.col('ids').alias('anchor_id'))
        .explode('anchor_id')
        # an entry none of whose members resolve to a molecule drops out entirely here
        .drop_nulls('anchor_id')
        .group_by('entry_id')
        .agg(pl.col('anchor_id').unique().alias('anchor_ids'))
    )

    cand = resolved.join(anchors, on='entry_id', how='inner').explode('anchor_ids').rename({'anchor_ids': 'anchor_id'})
    cand = cand.filter(~pl.col('ids').list.contains(pl.col('anchor_id')))

    pc = parent_child.rename({'id': 'anchor_id', 'related': 'pc_related'})
    cand = cand.join(pc, on='anchor_id', how='left')

    cand = cand.with_columns(
        status=pl
        .when(pl.col('ids').list.len() == 0)
        .then(pl.lit('NOVEL'))
        .when(pl.col('ids').list.set_intersection(pl.col('pc_related').fill_null([])).list.len() > 0)
        .then(pl.lit('PARENT_CHILD'))
        .otherwise(pl.lit('CONFLICT'))
    )

    return cand.select(
        pl.col('anchor_id').alias('id'),
        pl.col('member').alias('candidate'),
        'nct_id',
        'status',
    ).unique()


def _rewrite_and_reclassify_codes(
    cand: pl.DataFrame, name_index: pl.DataFrame, parent_child: pl.DataFrame
) -> pl.DataFrame:
    """Rule #8: rewrite descriptor phrases to their bare R&D code, then re-resolve.

    Rewriting e.g. ``akt inhibitor mk2206`` -> ``mk2206`` changes the candidate's
    identity, so its anchor-time status is stale. We re-resolve the rewritten candidate
    against ``name_index`` and reclassify:

    - drop it if it is now already a label of the anchor molecule (redundant)
    - drop it if the rewritten code is now over-ambiguous (> AMBIGUITY_CAP)
    - recompute NOVEL / PARENT_CHILD / CONFLICT so a code belonging to the anchor's
      parent/child family is marked PARENT_CHILD and dropped downstream

    Idempotent for candidates that are not rewritten (their resolution is unchanged
    from anchoring time).

    Args:
        cand: DataFrame[id, candidate, nct_id, status]
        name_index: DataFrame[name_norm, ids: list[str]]
        parent_child: DataFrame[id, related: list[str]]

    Returns:
        DataFrame[id, candidate, nct_id, status] with rewritten, reclassified candidates.
    """
    # rule #8: descriptor-wrapped code -> bare code (phrase has a class word AND a code)
    code = pl.col('candidate').str.extract(DRUG_CODE_REGEX, 0)
    cand = cand.with_columns(
        candidate=pl
        .when(code.is_not_null() & _has_class_keyword(pl.col('candidate')))
        .then(code)
        .otherwise(pl.col('candidate'))
    ).drop('status')

    # re-resolve the (possibly rewritten) candidate against the ChEMBL name index
    resolved = (
        cand
        .join(name_index, left_on='candidate', right_on='name_norm', how='left')
        .with_columns(ids=pl.col('ids').fill_null([]))
        .select('id', 'candidate', 'nct_id', 'ids')
    )

    # a rewritten code that is now over-ambiguous or already on the anchor molecule is
    # not a candidate for it
    resolved = resolved.filter(pl.col('ids').list.len() <= AMBIGUITY_CAP)
    resolved = resolved.filter(~pl.col('ids').list.contains(pl.col('id')))

    # reclassify status against the anchor molecule's parent/child family
    pc = parent_child.rename({'related': 'pc_related'})
    resolved = resolved.join(pc, on='id', how='left')

    return (
        resolved
        .with_columns(
            status=pl
            .when(pl.col('ids').list.len() == 0)
            .then(pl.lit('NOVEL'))
            .when(pl.col('ids').list.set_intersection(pl.col('pc_related').fill_null([])).list.len() > 0)
            .then(pl.lit('PARENT_CHILD'))
            .otherwise(pl.lit('CONFLICT'))
        )
        .select('id', 'candidate', 'nct_id', 'status')
        .unique()
    )


def _apply_cleanup_rules(
    cand: pl.DataFrame, regimen_index: pl.DataFrame, existing_per_id: pl.DataFrame
) -> pl.DataFrame:
    """Apply rules #5-#11 + drop PARENT_CHILD. Returns DataFrame[id, candidate, nct_id].

    Args:
        cand: DataFrame[id, candidate, nct_id, status]
        regimen_index: DataFrame[regimen_norm, ids: list[str]]
        existing_per_id: DataFrame[id, existing: list[str]]

    Returns:
        DataFrame[id, candidate, nct_id] with noise filtered out.
    """
    # drop PARENT_CHILD (keep NOVEL + CONFLICT). Descriptor-code extraction (#8) already
    # happened upstream in _rewrite_and_reclassify_codes, which also re-resolved the
    # rewritten code so PARENT_CHILD here reflects the bare code.
    cand = cand.filter(pl.col('status') != 'PARENT_CHILD')

    # #10: single-character
    cand = cand.filter(pl.col('candidate').str.len_chars() > 1)

    # #9: insulin units + any '%'
    cand = cand.filter(~pl.col('candidate').str.contains(r'^(u|gla)[- ]?\d{2,3}$'))
    cand = cand.filter(~pl.col('candidate').str.contains('%', literal=True))

    # #5: control noise
    cand = cand.filter(~pl.col('candidate').is_in(sorted(CONTROL_TERMS)))

    # #6: drug-class / cell-therapy keyword present, UNLESS the candidate is a bare
    # R&D code (descriptor phrases were already rewritten to their code upstream)
    cand = cand.filter(~_has_class_keyword(pl.col('candidate')) | pl.col('candidate').str.contains(DRUG_CODE_REGEX))

    # #7: regimen suppression (candidate equals a known regimen token)
    regimen_keys = regimen_index.select(pl.col('regimen_norm').alias('candidate')).unique()
    cand = cand.join(regimen_keys.with_columns(_is_regimen=pl.lit(True)), on='candidate', how='left')
    cand = cand.filter(pl.col('_is_regimen').is_null()).drop('_is_regimen')

    # #11: plural suppression (singular already on M)
    length = pl.col('candidate').str.len_chars()
    cand = cand.with_columns(
        singular=pl
        .when(pl.col('candidate').str.ends_with('ies'))
        .then(pl.col('candidate').str.slice(0, length - 3) + 'y')
        .when(pl.col('candidate').str.ends_with('es'))
        .then(pl.col('candidate').str.slice(0, length - 2))
        .when(pl.col('candidate').str.ends_with('s'))
        .then(pl.col('candidate').str.slice(0, length - 1))
        .otherwise(pl.col('candidate'))
    )
    cand = cand.join(existing_per_id, on='id', how='left')
    cand = cand.filter(
        (pl.col('singular') == pl.col('candidate'))
        | ~pl.col('existing').fill_null([]).list.contains(pl.col('singular'))
    ).drop('singular', 'existing')

    return cand.select('id', 'candidate', 'nct_id').unique()


def mine_aact_synonyms(mol_df: pl.DataFrame, entries: pl.DataFrame) -> pl.DataFrame:
    """Full AACT mining: anchor -> cleanup -> n_trials>=MIN_TRIALS -> DataFrame[id, label].

    The stored label is the normalized candidate string (v1: normalized form, which
    matches the anchor index; surface-form refinement is deferred).

    Args:
        mol_df: See :func:`_build_chembl_indexes`.
        entries: See :func:`parse_aact_entries`.

    Returns:
        DataFrame[id, label] of candidate AACT synonyms clearing MIN_TRIALS.
    """
    name_index, regimen_index, parent_child = _build_chembl_indexes(mol_df)

    # Per-molecule set of normalized existing names (name + synonym/tradeName labels),
    # used by rule #11 plural suppression. Intentionally parallels the label collection
    # in _build_chembl_indexes (different shape: grouped array vs exploded rows).
    existing_per_id = mol_df.select(
        'id',
        pl.concat_list(
            pl.concat_list(_normalize_name(pl.col('name'))),
            pl.concat_list(
                pl.col('synonyms').list.eval(_normalize_name(pl.element().struct.field('label'))),
                pl.col('tradeNames').list.eval(_normalize_name(pl.element().struct.field('label'))),
            ),
        ).alias('existing'),
    )

    anchored = _anchor_candidates(entries, name_index, parent_child)
    reclassified = _rewrite_and_reclassify_codes(anchored, name_index, parent_child)
    cleaned = _apply_cleanup_rules(reclassified, regimen_index, existing_per_id)

    return (
        cleaned
        .group_by('id', 'candidate')
        .agg(pl.col('nct_id').n_unique().alias('n_trials'))
        .filter(pl.col('n_trials') >= MIN_TRIALS)
        .select('id', pl.col('candidate').alias('label'))
    )


def merge_aact_synonyms(mol_combined: pl.DataFrame, aact_df: pl.DataFrame) -> pl.DataFrame:
    """Append AACT labels (deduped vs existing ChEMBL labels) as {label,'AACT'} structs.

    Args:
        mol_combined: Molecule DataFrame with id and synonyms (possibly null) columns.
        aact_df: DataFrame[id, label] as returned by :func:`mine_aact_synonyms`.

    Returns:
        ``mol_combined`` with ``synonyms`` carrying the fresh AACT labels appended.
        ``tradeNames`` is untouched: a trade name is a registered brand, and a label
        mined from trial free text has not been established to be one.
    """
    aact_grouped = aact_df.group_by('id').agg(pl.col('label').drop_nulls().unique().alias('aact_labels'))

    merged = mol_combined.join(aact_grouped, on='id', how='left').with_columns(
        synonyms_filled=pl.col('synonyms').fill_null([]),
        aact_labels_filled=pl.col('aact_labels').fill_null([]),
    )

    existing_lc = merged.select(
        'id',
        pl.col('synonyms_filled').list.eval(pl.element().struct.field('label').str.to_lowercase()).alias('existing_lc'),
    )

    fresh = (
        merged
        .select('id', 'aact_labels_filled')
        .explode('aact_labels_filled')
        .drop_nulls('aact_labels_filled')
        .join(existing_lc, on='id', how='left')
        .filter(~pl.col('existing_lc').list.contains(pl.col('aact_labels_filled').str.to_lowercase()))
        .select(
            'id',
            pl.struct(pl.col('aact_labels_filled').alias('label'), pl.lit(AACT_SOURCE).alias('source')).alias(
                'new_struct'
            ),
        )
        .group_by('id')
        .agg(pl.col('new_struct'))
    )

    return (
        merged
        .join(fresh, on='id', how='left')
        .with_columns(
            # array_union already dedups identical structs; list.unique here matches that.
            synonyms=pl
            .concat_list(pl.col('synonyms_filled'), pl.col('new_struct').fill_null([]))
            .list.unique(maintain_order=True)
            .list.sort()
        )
        .drop('aact_labels', 'synonyms_filled', 'aact_labels_filled', 'new_struct')
    )
