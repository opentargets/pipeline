"""Clinical-trial synonym mining for ChEMBL molecules.

Mines candidate drug synonyms from the OpenAI/AACT clinical-trial extraction and
anchors them to ChEMBL molecules. A PySpark port of the ``work/clinical_pairs/``
experiment, consumed by ``chembl_molecule`` to append ``source: "AACT"`` synonyms.

Pipeline: parse batch JSONL → normalized drug member sets → anchor against a
ChEMBL name index (with an ambiguity cap) → 11 cleanup rules → keep candidates
seen in ``MIN_TRIALS`` distinct trials → merge into the molecule synonyms.
"""

import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

from pts.pyspark.drug_utils.labels import (
    AACT_SOURCE,
    LABEL_SOURCE_SCHEMA,
    as_label_source,
)

# --- Tunables ---------------------------------------------------------------

AMBIGUITY_CAP = 10
MIN_TRIALS = 2

# --- Normalization ----------------------------------------------------------


def _normalize_name(col):
    """Lowercase, strip trademark symbols, trim, collapse internal whitespace."""
    stripped = f.regexp_replace(col, r'[®™©℠]', '')
    collapsed = f.regexp_replace(f.trim(stripped), r'\s+', ' ')
    return f.lower(collapsed)


# --- Batch parsing ----------------------------------------------------------

_DRUG_LIST_SCHEMA = ArrayType(
    StructType([
        StructField('drug', StringType()),
        StructField('synonyms', ArrayType(StringType())),
    ])
)

BATCH_INNER_SCHEMA = StructType([
    StructField('investigated_drugs', _DRUG_LIST_SCHEMA),
    StructField('comparator_drugs', _DRUG_LIST_SCHEMA),
    StructField('supportive_drugs', _DRUG_LIST_SCHEMA),
])


def parse_aact_batch(batch_raw):
    """Parse OpenAI batch output into one row per drug entry with a normalized member set.

    Returns DataFrame[nct_id, members: array<string>] (normalized, deduped, non-empty).
    """
    messages = (
        batch_raw
        .select(
            f.col('custom_id').alias('nct_id'),
            f.explode('response.body.output').alias('out'),
        )
        .filter(f.col('out.type') == 'message')
        .select('nct_id', f.explode('out.content').alias('content'))
        # content.text is itself a JSON string (OpenAI structured output is double-encoded);
        # decode it into BATCH_INNER_SCHEMA.
        .select(
            'nct_id',
            f.from_json(f.col('content.text'), BATCH_INNER_SCHEMA).alias('parsed'),
        )
    )

    roles = f.array_union(
        f.array_union(
            f.coalesce(f.col('parsed.investigated_drugs'), f.array().cast(_DRUG_LIST_SCHEMA)),
            f.coalesce(f.col('parsed.comparator_drugs'), f.array().cast(_DRUG_LIST_SCHEMA)),
        ),
        f.coalesce(f.col('parsed.supportive_drugs'), f.array().cast(_DRUG_LIST_SCHEMA)),
    )

    return (
        messages
        .withColumn('entry', f.explode(roles))
        .withColumn(
            'members',
            f.array_union(
                f.array(f.col('entry.drug')),
                f.coalesce(f.col('entry.synonyms'), f.array().cast('array<string>')),
            ),
        )
        .withColumn(
            'members',
            f.array_distinct(
                f.filter(
                    f.transform(f.col('members'), _normalize_name),
                    lambda m: (m.isNotNull()) & (f.length(m) > 0),
                )
            ),
        )
        .filter(f.size('members') > 0)
        .select('nct_id', 'members')
    )


# --- ChEMBL anchor indexes --------------------------------------------------


def _build_chembl_indexes(mol_df):
    """Build (name_index, regimen_index, parent_child) from ChEMBL-source names.

    name_index:    DataFrame[name_norm, ids: array<string>]
    regimen_index: DataFrame[regimen_norm, ids: array<string>]  (suppression only)
    parent_child:  DataFrame[id, related: array<string>]  (parent + children)
    """
    empty_ls = f.array().cast(LABEL_SOURCE_SCHEMA)
    labels = (
        mol_df
        .select(
            'id',
            f.array_union(
                f.array(f.col('name')),
                f.array_union(
                    f.transform(
                        f.coalesce(f.col('synonyms'), empty_ls),
                        lambda s: s['label'],
                    ),
                    f.transform(
                        f.coalesce(f.col('tradeNames'), empty_ls),
                        lambda t: t['label'],
                    ),
                ),
            ).alias('labels'),
        )
        .select('id', f.explode('labels').alias('label'))
        .withColumn('name_norm', _normalize_name(f.col('label')))
        .filter(f.length('name_norm') > 0)
    )

    name_index = labels.groupBy('name_norm').agg(f.collect_set('id').alias('ids'))

    # "<ingredient> COMPONENT OF <regimen>" -> regimen token (normalized text is lowercased)
    regimen_index = (
        labels
        .withColumn(
            'regimen_norm',
            f.regexp_extract(f.col('name_norm'), r'\bcomponent of\s+(.+)$', 1),
        )
        .filter(f.length('regimen_norm') > 0)
        .groupBy('regimen_norm')
        .agg(f.collect_set('id').alias('ids'))
    )

    empty_str_arr = f.array().cast('array<string>')
    children = mol_df.select(
        'id',
        f.coalesce(f.col('childChemblIds'), empty_str_arr).alias('related'),
    )
    parents = mol_df.filter(f.col('parentId').isNotNull()).select('id', f.array(f.col('parentId')).alias('related'))
    parent_child = (
        children
        .union(parents)
        .groupBy('id')
        .agg(f.array_distinct(f.flatten(f.collect_list('related'))).alias('related'))
    )

    return name_index, regimen_index, parent_child


# --- Anchoring --------------------------------------------------------------


def _anchor_candidates(entries, name_index, parent_child):
    """Anchor member sets to molecules and emit (id, candidate, nct_id, status).

    For each trial drug entry (a normalized member set), resolve members against
    name_index to find which ChEMBL molecule(s) the entry anchors to, then emit
    each member that is NOT already on an anchored molecule as a candidate
    synonym, classified by status.

    Entries where any single member resolves to more than AMBIGUITY_CAP molecules
    are dropped entirely.

    Note: the same (id, candidate, nct_id) may appear with more than one status
    when a trial contributes multiple drug entries; downstream trial counting
    must use COUNT(DISTINCT nct_id).

    Args:
        entries: DataFrame[nct_id, members: array<string>]
        name_index: DataFrame[name_norm, ids: array<string>]
        parent_child: DataFrame[id, related: array<string>]

    Returns:
        DataFrame[id, candidate, nct_id, status] where id is an anchored
        molecule, candidate is a member not already on id, and status is one of
        NOVEL / PARENT_CHILD / CONFLICT.
    """
    # Deterministic per-entry key (nct_id + sorted member set). Avoids
    # f.monotonically_increasing_id(), which is nondeterministic across
    # re-evaluations and would let the `poisoned` and `anchors` branches below
    # see inconsistent ids under Spark adaptive re-planning. The control-char
    # separators cannot occur in normalized names or NCT ids, so no collision.
    entries = entries.withColumn(
        'entry_id',
        f.sha2(
            f.concat_ws('', f.col('nct_id'), f.array_join(f.array_sort(f.col('members')), '')),
            256,
        ),
    )

    members = entries.select('entry_id', 'nct_id', f.explode('members').alias('member'))

    # resolve each member against name_index; unresolved -> empty ids array
    resolved = members.join(name_index, members['member'] == name_index['name_norm'], 'left').select(
        'entry_id',
        'nct_id',
        'member',
        f.coalesce(f.col('ids'), f.array().cast('array<string>')).alias('ids'),
    )

    # poison: drop any entry where a single member resolves to > AMBIGUITY_CAP molecules
    poisoned = (
        resolved
        .groupBy('entry_id')
        .agg(f.max(f.size('ids')).alias('max_ids'))
        .filter(f.col('max_ids') > AMBIGUITY_CAP)
        .select('entry_id')
    )
    resolved = resolved.join(poisoned, on='entry_id', how='left_anti')

    # collect the union of all resolved molecule ids per entry (anchor set)
    anchors = (
        resolved
        .select('entry_id', f.explode('ids').alias('anchor_id'))
        .groupBy('entry_id')
        .agg(f.collect_set('anchor_id').alias('anchor_ids'))
    )

    # cross each member with each anchored molecule of its entry
    cand = resolved.join(anchors, on='entry_id', how='inner')
    cand = cand.withColumn('anchor_id', f.explode('anchor_ids'))

    # drop members already belonging to the anchor molecule (not candidates for it)
    cand = cand.filter(~f.array_contains(f.col('ids'), f.col('anchor_id')))

    # join parent_child info for the anchor to determine status
    pc = parent_child.withColumnRenamed('id', 'anchor_id').withColumnRenamed('related', 'pc_related')
    cand = cand.join(pc, on='anchor_id', how='left')

    empty_str_arr = f.array().cast('array<string>')
    cand = cand.withColumn(
        'status',
        f
        .when(f.size('ids') == 0, f.lit('NOVEL'))
        .when(
            f.arrays_overlap(f.col('ids'), f.coalesce(f.col('pc_related'), empty_str_arr)),
            f.lit('PARENT_CHILD'),
        )
        .otherwise(f.lit('CONFLICT')),
    )

    return cand.select(
        f.col('anchor_id').alias('id'),
        f.col('member').alias('candidate'),
        'nct_id',
        'status',
    ).distinct()


# --- Cleanup rules ----------------------------------------------------------

# v1 port of the experiment's cleanup blacklists — expected to grow with corpus coverage.
CODE_REGEX = r'\b[a-z]{1,6}-?\d{3,}[a-z0-9]*\b'

# v1 port of the experiment's cleanup blacklists — expected to grow with corpus coverage.
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
# v1 port of the experiment's cleanup blacklists — expected to grow with corpus coverage.
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


def _has_class_keyword(col):
    """True when the candidate text contains any drug-class / cell-therapy keyword as a whole word."""
    return col.rlike(_CLASS_PATTERN)


def _rewrite_and_reclassify_codes(cand, name_index, parent_child):
    """Rule #8: rewrite descriptor phrases to their bare R&D code, then re-resolve.

    Rewriting e.g. ``akt inhibitor mk2206`` -> ``mk2206`` changes the candidate's
    identity, so its anchor-time status is stale. We re-resolve the rewritten
    candidate against ``name_index`` and reclassify:

    - drop it if it is now already a label of the anchor molecule (redundant)
    - drop it if the rewritten code is now over-ambiguous (> AMBIGUITY_CAP)
    - recompute NOVEL / PARENT_CHILD / CONFLICT so a code belonging to the
      anchor's parent/child family is marked PARENT_CHILD and dropped downstream

    Idempotent for candidates that are not rewritten (their resolution is
    unchanged from anchoring time).

    Args:
        cand: DataFrame[id, candidate, nct_id, status]
        name_index: DataFrame[name_norm, ids: array<string>]
        parent_child: DataFrame[id, related: array<string>]

    Returns:
        DataFrame[id, candidate, nct_id, status] with rewritten, reclassified candidates.
    """
    # rule #8: descriptor-wrapped code -> bare code (phrase has a class word AND a code)
    cand = cand.withColumn('code', f.regexp_extract(f.col('candidate'), CODE_REGEX, 0))
    cand = cand.withColumn(
        'candidate',
        f.when(
            (f.length('code') > 0) & _has_class_keyword(f.col('candidate')),
            f.col('code'),
        ).otherwise(f.col('candidate')),
    ).drop('code', 'status')

    # re-resolve the (possibly rewritten) candidate against the ChEMBL name index
    resolved = cand.join(name_index, cand['candidate'] == name_index['name_norm'], 'left').select(
        'id',
        'candidate',
        'nct_id',
        f.coalesce(f.col('ids'), f.array().cast('array<string>')).alias('ids'),
    )

    # a rewritten code that is now over-ambiguous or already on the anchor molecule
    # is not a candidate for it
    resolved = resolved.filter(f.size('ids') <= AMBIGUITY_CAP)
    resolved = resolved.filter(~f.array_contains(f.col('ids'), f.col('id')))

    # reclassify status against the anchor molecule's parent/child family
    pc = parent_child.withColumnRenamed('related', 'pc_related')
    resolved = resolved.join(pc, on='id', how='left')
    empty_str_arr = f.array().cast('array<string>')
    return (
        resolved
        .withColumn(
            'status',
            f
            .when(f.size('ids') == 0, f.lit('NOVEL'))
            .when(
                f.arrays_overlap(f.col('ids'), f.coalesce(f.col('pc_related'), empty_str_arr)),
                f.lit('PARENT_CHILD'),
            )
            .otherwise(f.lit('CONFLICT')),
        )
        .select('id', 'candidate', 'nct_id', 'status')
        .distinct()
    )


def _apply_cleanup_rules(cand, regimen_index, existing_per_id):
    """Apply rules #5-#11 + drop PARENT_CHILD. Returns DataFrame[id, candidate, nct_id].

    Args:
        cand: DataFrame[id, candidate, nct_id, status]
        regimen_index: DataFrame[regimen_norm, ids: array<string>]
        existing_per_id: DataFrame[id, existing: array<string>]

    Returns:
        DataFrame[id, candidate, nct_id] with noise filtered out.
    """
    # drop PARENT_CHILD (keep NOVEL + CONFLICT). Descriptor-code extraction (#8)
    # already happened upstream in _rewrite_and_reclassify_codes, which also
    # re-resolved the rewritten code so PARENT_CHILD here reflects the bare code.
    cand = cand.filter(f.col('status') != 'PARENT_CHILD')

    # #10: single-character
    cand = cand.filter(f.length('candidate') > 1)

    # #9: insulin units + any '%'
    cand = cand.filter(~f.col('candidate').rlike(r'^(u|gla)[- ]?\d{2,3}$'))
    cand = cand.filter(~f.col('candidate').contains('%'))

    # #5: control noise
    control_array = f.array([f.lit(t) for t in sorted(CONTROL_TERMS)])
    cand = cand.filter(~f.array_contains(control_array, f.col('candidate')))

    # #6: drug-class / cell-therapy keyword present, UNLESS the candidate is a bare
    # R&D code (descriptor phrases were already rewritten to their code upstream)
    cand = cand.filter(~_has_class_keyword(f.col('candidate')) | f.col('candidate').rlike(CODE_REGEX))

    # #7: regimen suppression (candidate equals a known regimen token)
    regimen_keys = regimen_index.select(f.col('regimen_norm').alias('candidate')).distinct()
    cand = cand.join(regimen_keys.withColumn('_is_regimen', f.lit(True)), on='candidate', how='left')
    cand = cand.filter(f.col('_is_regimen').isNull()).drop('_is_regimen')

    # #11: plural suppression (singular already on M)
    cand = cand.withColumn(
        'singular',
        f
        .when(
            f.col('candidate').endswith('ies'),
            f.concat(f.expr('left(candidate, length(candidate) - 3)'), f.lit('y')),
        )
        .when(
            f.col('candidate').endswith('es'),
            f.expr('left(candidate, length(candidate) - 2)'),
        )
        .when(
            f.col('candidate').endswith('s'),
            f.expr('left(candidate, length(candidate) - 1)'),
        )
        .otherwise(f.col('candidate')),
    )
    cand = cand.join(existing_per_id, on='id', how='left')
    cand = cand.filter(
        (f.col('singular') == f.col('candidate'))
        | ~f.array_contains(
            f.coalesce(f.col('existing'), f.array().cast('array<string>')),
            f.col('singular'),
        )
    ).drop('singular', 'existing')

    return cand.select('id', 'candidate', 'nct_id').distinct()


# --- Orchestration ----------------------------------------------------------


def mine_aact_synonyms(mol_df, entries):
    """Full AACT mining: anchor -> cleanup -> n_trials>=MIN_TRIALS -> DataFrame[id, label].

    The stored label is the normalized candidate string (v1: normalized form, which
    matches the anchor index; surface-form refinement is deferred).
    """
    name_index, regimen_index, parent_child = _build_chembl_indexes(mol_df)

    # Per-molecule set of normalized existing names (name + synonym/tradeName labels),
    # used by rule #11 plural suppression. Intentionally parallels the label collection
    # in _build_chembl_indexes (different shape: grouped array vs exploded rows).
    empty_ls = f.array().cast(LABEL_SOURCE_SCHEMA)
    existing_per_id = mol_df.select(
        'id',
        f.array_union(
            f.array(_normalize_name(f.col('name'))),
            f.array_union(
                f.transform(
                    f.coalesce(f.col('synonyms'), empty_ls),
                    lambda s: _normalize_name(s['label']),
                ),
                f.transform(
                    f.coalesce(f.col('tradeNames'), empty_ls),
                    lambda t: _normalize_name(t['label']),
                ),
            ),
        ).alias('existing'),
    )

    anchored = _anchor_candidates(entries, name_index, parent_child)
    reclassified = _rewrite_and_reclassify_codes(anchored, name_index, parent_child)
    cleaned = _apply_cleanup_rules(reclassified, regimen_index, existing_per_id)

    return (
        cleaned
        .groupBy('id', 'candidate')
        .agg(f.countDistinct('nct_id').alias('n_trials'))
        .filter(f.col('n_trials') >= MIN_TRIALS)
        .select('id', f.col('candidate').alias('label'))
    )


def merge_aact_synonyms(mol_combined, aact_df):
    """Append AACT labels (deduped vs existing ChEMBL labels) as {label,'AACT'} structs."""
    aact_grouped = aact_df.groupBy('id').agg(f.collect_set('label').alias('aact_labels'))

    merged = mol_combined.join(aact_grouped, on='id', how='left')

    existing_lc = f.transform(
        f.coalesce(f.col('synonyms'), f.array().cast(LABEL_SOURCE_SCHEMA)),
        lambda s: f.lower(s['label']),
    )
    fresh = f.filter(
        f.coalesce(f.col('aact_labels'), f.array().cast('array<string>')),
        lambda c: ~f.array_contains(existing_lc, f.lower(c)),
    )
    new_structs = f.transform(fresh, lambda c: as_label_source(c, AACT_SOURCE))

    return merged.withColumn(
        'synonyms',
        # array_sort for deterministic output; array_union already dedups identical structs.
        f.array_sort(
            f.array_union(
                f.coalesce(f.col('synonyms'), f.array().cast(LABEL_SOURCE_SCHEMA)),
                new_structs,
            )
        ).cast(LABEL_SOURCE_SCHEMA),
    ).drop('aact_labels')
