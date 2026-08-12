"""ChEMBL Molecule processing.

Processes raw ChEMBL molecule tables into the Open Targets molecule format,
including synonyms, DrugBank cross-references, and molecule hierarchy. ChEMBL's
own cross-reference table is deliberately not joined here (see
:func:`_process_molecule_cross_references`).

Clinical-trial (AACT) synonym mining lives in this module too (``_parse_aact_entries``
onward) -- it moved here from the ``pts.pyspark.drug_utils.aact_synonyms`` port, since
it only ever runs as part of this step. ``parse_batch_results`` (from ``clinical_mining``,
already used by ``clinical_report``) reads the OpenAI batch output straight into polars;
everything downstream of it -- normalizing candidate names, anchoring them to ChEMBL
molecules, and the eleven cleanup rules -- is a field-for-field polars port of that module.
"""

from pathlib import Path
from typing import Any

import polars as pl
from clinical_mining.data_sources.aact.llm_extractor import parse_batch_results
from clinical_mining.schemas import ClinicalSource
from loguru import logger
from otter.config.model import Config

from pts.postgres import read_dump_tables

SCHEMA_NAME = 'public'
"""Schema the ChEMBL tables live in inside the restored dump."""

TABLES = {
    'molecule_dictionary': ['molregno', 'chembl_id', 'pref_name', 'molecule_type'],
    'compound_structures': ['molregno', 'canonical_smiles', 'standard_inchi_key', 'molfile'],
    'molecule_hierarchy': ['molregno', 'parent_molregno'],
    'molecule_synonyms': ['molsyn_id', 'molregno', 'synonyms', 'syn_type'],
}
"""ChEMBL tables and columns this step needs, restored from the dump."""

CHEMBL_SOURCE = 'ChEMBL'
AACT_SOURCE = ClinicalSource.AACT.value

# --- AACT synonym mining tunables --------------------------------------------

AMBIGUITY_CAP = 10
MIN_TRIALS = 2

# v1 port of the experiment's cleanup blacklists -- expected to grow with corpus coverage.
CODE_REGEX = r'\b[a-z]{1,6}-?\d{3,}[a-z0-9]*\b'

# v1 port of the experiment's cleanup blacklists -- expected to grow with corpus coverage.
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
# v1 port of the experiment's cleanup blacklists -- expected to grow with corpus coverage.
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


def chembl_molecule(
    source: dict[str, Path],
    destination: Path,
    _settings: dict[str, Any],
    _config: Config,
) -> None:
    """Transform raw ChEMBL molecule tables into the Open Targets molecule format.

    Args:
        source: Dictionary with paths to:
            - chembl: Path to the ChEMBL ``pg_dump`` archive.
            - drugbank: DrugBank-to-ChEMBL id mapping (tab-separated, ``.csv.gz``).
            - aact_extraction_batch_results: (optional) OpenAI batch output for
              clinical-trial synonym mining; when present, AACT synonyms are
              appended to the molecules.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        _config: Config object (not used).
    """
    logger.info(f'Restoring {list(TABLES)} from {source["chembl"]}')
    tables = read_dump_tables(str(source['chembl']), TABLES, schema_name=SCHEMA_NAME)

    logger.info(f'Reading drugbank lookup from {source["drugbank"]}')
    drugbank_lookup = pl.read_csv(source['drugbank'], separator='\t')

    aact_batch = None
    if 'aact_extraction_batch_results' in source:
        logger.info(f'Reading AACT batch results from {source["aact_extraction_batch_results"]}')
        aact_batch = parse_batch_results(str(source['aact_extraction_batch_results']))

    logger.info('Processing molecules')
    output_df = process_molecules(
        tables['molecule_dictionary'],
        tables['compound_structures'],
        tables['molecule_hierarchy'],
        tables['molecule_synonyms'],
        drugbank_lookup,
        aact_batch,
    )

    logger.info(f'Writing molecules to {destination}')
    output_df.write_parquet(destination, mkdir=True)


def process_molecules(
    molecule_dictionary: pl.DataFrame,
    compound_structures: pl.DataFrame,
    molecule_hierarchy: pl.DataFrame,
    molecule_synonyms: pl.DataFrame,
    drugbank_lookup: pl.DataFrame,
    aact_batch: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Build ChEMBL molecules by joining raw ChEMBL tables.

    Args:
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        compound_structures: Raw ChEMBL compound_structures table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        molecule_synonyms: Raw ChEMBL molecule_synonyms table.
        drugbank_lookup: Drugbank to ChEMBL id mapping, as read from the raw file.
        aact_batch: (optional) Parsed AACT batch extractions, as returned by
            :func:`clinical_mining.data_sources.aact.llm_extractor.parse_batch_results`.
            When provided, AACT synonyms are appended (deduped case-insensitively vs
            existing ChEMBL labels) before the final name-coalesce so that AACT labels
            never become the molecule name.

    Returns:
        Processed molecule DataFrame with the eleven output columns.
    """
    # Prepare drugbank lookup -- rename columns to match expected format
    drugbank = drugbank_lookup.select(
        pl.col("From src:'1'").alias('id'),
        pl.col("To src:'2'").alias('drugbank_id'),
    )

    mols = _molecule_preprocess(
        molecule_dictionary,
        compound_structures,
        molecule_hierarchy,
        molecule_synonyms,
        drugbank,
    )

    synonyms = _process_molecule_synonyms(mols)
    cross_references = _process_molecule_cross_references(mols)
    hierarchy = _process_molecule_hierarchy(mols)

    mol_combined = (
        mols.drop('syns')
        .join(synonyms, on='id', how='left')
        .join(cross_references, on='id', how='left')
        .join(hierarchy, on='id', how='left')
    )

    # Optionally mine and merge AACT synonyms BEFORE the name-coalesce so that
    # AACT labels are never selected as the molecule name.
    if aact_batch is not None:
        entries = _parse_aact_entries(aact_batch)
        # mine_aact_synonyms / _build_chembl_indexes expect non-null arrays; fill here.
        mol_for_index = mol_combined.select(
            'id',
            'name',
            pl.col('synonyms').fill_null([]),
            pl.col('tradeNames').fill_null([]),
            'parentId',
            pl.col('childChemblIds').fill_null([]),
        )
        aact_df = mine_aact_synonyms(mol_for_index, entries)
        mol_combined = merge_aact_synonyms(mol_combined, aact_df)

    # Final processing -- ensure name is populated and deduplicate. childChemblIds is
    # deliberately NOT coalesced here: the reference leaves it null for a molecule with
    # no children (only synonyms/tradeNames are coalesced to []), and the vast majority
    # of molecules have no children, so filling it would change a published column
    # across nearly the whole dataset. The AACT-branch fill above is a different,
    # correct case: mine_aact_synonyms/_build_chembl_indexes need a non-null array to
    # index over, matching what the pyspark reference does for its own index input.
    return (
        mol_combined.with_columns(
            synonyms=pl.col('synonyms').fill_null([]),
            tradeNames=pl.col('tradeNames').fill_null([]),
        )
        .with_columns(
            name=pl.coalesce(
                pl.col('name'),
                pl.col('synonyms')
                .list.eval(pl.element().filter(pl.element().struct.field('source') == CHEMBL_SOURCE))
                .list.first()
                .struct.field('label'),
                pl.col('id'),
            )
        )
        .unique(subset=['id'])
        .select(
            'id',
            'canonicalSmiles',
            'inchiKey',
            'molblock',
            'drugType',
            'name',
            'parentId',
            'childChemblIds',
            'synonyms',
            'tradeNames',
            'crossReferences',
        )
    )


def _molecule_preprocess(
    molecule_dictionary: pl.DataFrame,
    compound_structures: pl.DataFrame,
    molecule_hierarchy: pl.DataFrame,
    molecule_synonyms: pl.DataFrame,
    drugbank: pl.DataFrame,
) -> pl.DataFrame:
    """Preprocess raw ChEMBL molecule tables into one row per molecule.

    Args:
        molecule_dictionary: Raw ChEMBL molecule_dictionary table.
        compound_structures: Raw ChEMBL compound_structures table.
        molecule_hierarchy: Raw ChEMBL molecule_hierarchy table.
        molecule_synonyms: Raw ChEMBL molecule_synonyms table.
        drugbank: Drugbank lookup table (id, drugbank_id).

    Returns:
        Preprocessed molecule DataFrame.
    """
    parent = (
        molecule_hierarchy.join(
            molecule_dictionary.select(
                pl.col('molregno').alias('parent_molregno'),
                pl.col('chembl_id').alias('parentId'),
            ),
            on='parent_molregno',
            how='left',
        ).select('molregno', 'parentId')
    )

    # One struct array per molecule. The ordering is determinism-in-principle only
    # and does not reach the output: `syns` is dropped in `process_molecules` above,
    # and its only consumer, `_process_molecule_synonyms`, explodes it into two set
    # aggregations where order cannot survive.
    synonyms = molecule_synonyms.group_by('molregno').agg(
        pl.struct('molsyn_id', 'synonyms', 'syn_type').alias('syns')
    )

    return (
        molecule_dictionary.rename({'chembl_id': 'id'})
        .join(compound_structures, on='molregno', how='left')
        .join(parent, on='molregno', how='left')
        .join(synonyms, on='molregno', how='left')
        .select(
            'id',
            pl.col('canonical_smiles').alias('canonicalSmiles'),
            pl.col('standard_inchi_key').alias('inchiKey'),
            # ChEMBL ships some molfile values as a full SD-file record (molblock +
            # appended SDF property tags); truncate to the bare molblock by dropping
            # everything after the `M  END` terminator. If `M  END` is absent the
            # string is left unchanged.
            pl.col('molfile')
            .str.replace_all(
                # the terminator may or may not be followed by a newline: the
                # relational column is inconsistent, unlike the ES value which always
                # had one before its SDF property block. Emit exactly one either way --
                # all 2,897,819 molblocks in the 26.06 release end "M  END\n", so
                # dropping it would change a published column.
                r'(?s)(\nM  END)\n?.*',
                '$1\n',
            )
            .alias('molblock'),
            pl.col('molecule_type').fill_null('Unknown').alias('drugType'),
            # str.strip_chars() with no argument strips all Unicode whitespace; Spark's
            # trim() strips only the ASCII space. Pinned to ' ' explicitly so this
            # matches the reference rather than happening to agree on today's data.
            pl.col('pref_name').str.strip_chars(' ').alias('name'),
            'parentId',
            'syns',
        )
        # Remove circular references
        .with_columns(
            parentId=pl.when(pl.col('parentId') == pl.col('id')).then(None).otherwise(pl.col('parentId'))
        )
        .join(drugbank, on='id', how='left')
    )


def _process_molecule_synonyms(preprocessed_mols: pl.DataFrame) -> pl.DataFrame:
    """Group synonyms into sets of trade names and other synonyms.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id, tradeNames, and synonyms columns ({label, source} structs).
    """
    synonyms = (
        preprocessed_mols.select('id', 'syns')
        .explode('syns')
        # A molecule with no molecule_synonyms rows at all joins to a null `syns`
        # array. pyspark's default `explode` drops that row entirely; polars' keeps
        # it as a null row instead, so it is dropped explicitly here.
        .drop_nulls('syns')
        .unnest('syns')
        .with_columns(
            syn_type=pl.col('syn_type').str.to_uppercase(),
            synonym=pl.col('synonyms'),
        )
    )

    trade_names = (
        synonyms.filter(pl.col('syn_type') == 'TRADE_NAME')
        .group_by('id')
        # collect_set on the pyspark side drops nulls and dedups. polars' bare list
        # aggregation does neither, so a NULL `synonyms` text (a molecule_synonyms row
        # with no label) would otherwise survive as `None` inside the array instead of
        # being dropped.
        .agg(pl.col('synonym').drop_nulls().unique().alias('_trade'))
    )

    other_synonyms = (
        synonyms.filter(pl.col('syn_type') != 'TRADE_NAME')
        .group_by('id')
        .agg(pl.col('synonym').drop_nulls().unique().alias('_syn'))
    )

    full = trade_names.join(other_synonyms, on='id', how='full', coalesce=True)

    return full.with_columns(
        synonyms=pl.col('_syn')
        .fill_null([])
        .list.eval(pl.struct(pl.element().alias('label'), pl.lit(CHEMBL_SOURCE).alias('source')))
        .list.sort(),
        tradeNames=pl.col('_trade')
        .fill_null([])
        .list.eval(pl.struct(pl.element().alias('label'), pl.lit(CHEMBL_SOURCE).alias('source')))
        .list.sort(),
    ).drop('_syn', '_trade')


def _process_molecule_hierarchy(preprocessed_mols: pl.DataFrame) -> pl.DataFrame:
    """Group all child molecules by parent chembl_id.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and childChemblIds columns.
    """
    return (
        preprocessed_mols.select('id', 'parentId')
        .filter(pl.col('id') != pl.col('parentId'))
        .filter(pl.col('parentId').is_not_null())
        .group_by('parentId')
        .agg(pl.col('id').drop_nulls().unique().alias('childChemblIds'))
        .rename({'parentId': 'id'})
    )


def _process_molecule_cross_references(preprocessed_mols: pl.DataFrame) -> pl.DataFrame:
    """Group DrugBank cross references for each molecule id.

    ChEMBL's own cross-reference table is not joined here. Rebuilding it from the
    raw relational dump is unreliable (the best measured rule reaches 44% precision),
    so it was dropped rather than fabricated. DrugBank is unaffected, since it comes
    from a separate input.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.

    Returns:
        DataFrame with id and crossReferences columns.
    """
    return _process_singleton_cross_references(preprocessed_mols, 'drugbank_id', 'drugbank')


def _process_singleton_cross_references(
    preprocessed_mols: pl.DataFrame,
    reference_id_column: str,
    source: str,
) -> pl.DataFrame:
    """Process singleton cross references (e.g., drugbank_id) into a crossReferences column.

    Args:
        preprocessed_mols: Preprocessed molecule DataFrame.
        reference_id_column: Column name containing the reference id.
        source: Name of the source for the cross reference.

    Returns:
        DataFrame with id and crossReferences (array<struct<source,ids>>) columns.
    """
    return (
        preprocessed_mols.filter(pl.col(reference_id_column).is_not_null())
        .select('id', pl.col(reference_id_column).cast(pl.Utf8))
        .group_by('id')
        # collect_set drops nulls and dedups; the pre-filter above already removes
        # nulls, but drop_nulls().unique() is kept to match the site's contract
        # explicitly rather than relying on the filter alone.
        .agg(pl.col(reference_id_column).drop_nulls().unique().alias('ids'))
        .with_columns(
            crossReferences=pl.concat_list(pl.struct(pl.lit(source).alias('source'), pl.col('ids')))
        )
        .select('id', 'crossReferences')
    )


# --- AACT synonym mining -----------------------------------------------------
#
# Mines candidate drug synonyms from the OpenAI/AACT clinical-trial extraction and
# anchors them to ChEMBL molecules. A polars port of the pyspark
# `pts.pyspark.drug_utils.aact_synonyms` module (itself a port of the
# `work/clinical_pairs/` experiment).
#
# Pipeline: parse batch output (via clinical_mining's `parse_batch_results`) -> normalized
# drug member sets -> anchor against a ChEMBL name index (with an ambiguity cap) -> eleven
# cleanup rules -> keep candidates seen in `MIN_TRIALS` distinct trials -> merge into the
# molecule synonyms.


def _normalize_name(expr: pl.Expr) -> pl.Expr:
    """Lowercase, strip trademark symbols, trim, collapse internal whitespace."""
    stripped = expr.str.replace_all(r'[®™©℠]', '')
    collapsed = stripped.str.strip_chars().str.replace_all(r'\s+', ' ')
    return collapsed.str.to_lowercase()


def _has_class_keyword(expr: pl.Expr) -> pl.Expr:
    """True when the candidate text contains any drug-class / cell-therapy keyword as a whole word."""
    return expr.str.contains(_CLASS_PATTERN)


def _parse_aact_entries(batch: pl.DataFrame) -> pl.DataFrame:
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
        batch.select(pl.col('id').alias('nct_id'), roles.alias('entry'))
        .explode('entry')
        # pyspark's default `explode` drops rows with an empty or null array; polars'
        # keeps a null row for both cases, so it is dropped explicitly here.
        .drop_nulls('entry')
        .unnest('entry')
        .with_columns(
            members=pl.concat_list(pl.concat_list(pl.col('drug')), pl.col('synonyms').fill_null([]))
        )
        .with_columns(
            members=pl.col('members')
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
        mol_df.select(
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
        labels.with_columns(regimen_norm=pl.col('name_norm').str.extract(r'\bcomponent of\s+(.+)$', 1))
        .filter(pl.col('regimen_norm').is_not_null() & (pl.col('regimen_norm').str.len_chars() > 0))
        .group_by('regimen_norm')
        .agg(pl.col('id').drop_nulls().unique().alias('ids'))
    )

    children = mol_df.select('id', pl.col('childChemblIds').alias('related'))
    parents = mol_df.filter(pl.col('parentId').is_not_null()).select(
        'id', pl.concat_list(pl.col('parentId')).alias('related')
    )
    parent_child = (
        pl.concat([children, parents])
        .group_by('id')
        .agg(
            pl.col('related').list.explode(keep_nulls=False, empty_as_null=False).unique(maintain_order=True).alias(
                'related'
            )
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
    # entries cannot collide onto the same key -- unlike pyspark, polars has no
    # adaptive-replanning hazard to guard against, so a plain deterministic string
    # (rather than a hash) is enough here.
    entries = entries.with_columns(
        entry_id=pl.col('nct_id') + pl.lit('\x1f') + pl.col('members').list.sort().list.join('\x1f')
    )

    members = entries.select('entry_id', 'nct_id', pl.col('members').alias('member')).explode('member')

    resolved = (
        members.join(name_index, left_on='member', right_on='name_norm', how='left')
        .with_columns(ids=pl.col('ids').fill_null([]))
        .select('entry_id', 'nct_id', 'member', 'ids')
    )

    poisoned = (
        resolved.group_by('entry_id')
        .agg(pl.col('ids').list.len().max().alias('max_ids'))
        .filter(pl.col('max_ids') > AMBIGUITY_CAP)
        .select('entry_id')
    )
    resolved = resolved.join(poisoned, on='entry_id', how='anti')

    anchors = (
        resolved.select('entry_id', pl.col('ids').alias('anchor_id'))
        .explode('anchor_id')
        # collect_set drops nulls and dedups; an entry whose every member resolves to
        # zero molecules contributes no rows here at all, matching pyspark's explode
        # of an empty array.
        .drop_nulls('anchor_id')
        .group_by('entry_id')
        .agg(pl.col('anchor_id').unique().alias('anchor_ids'))
    )

    cand = resolved.join(anchors, on='entry_id', how='inner').explode('anchor_ids').rename({'anchor_ids': 'anchor_id'})
    cand = cand.filter(~pl.col('ids').list.contains(pl.col('anchor_id')))

    pc = parent_child.rename({'id': 'anchor_id', 'related': 'pc_related'})
    cand = cand.join(pc, on='anchor_id', how='left')

    cand = cand.with_columns(
        status=pl.when(pl.col('ids').list.len() == 0)
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
    code = pl.col('candidate').str.extract(CODE_REGEX, 0)
    cand = cand.with_columns(
        candidate=pl.when(code.is_not_null() & _has_class_keyword(pl.col('candidate'))).then(code).otherwise(
            pl.col('candidate')
        )
    ).drop('status')

    # re-resolve the (possibly rewritten) candidate against the ChEMBL name index
    resolved = (
        cand.join(name_index, left_on='candidate', right_on='name_norm', how='left')
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
        resolved.with_columns(
            status=pl.when(pl.col('ids').list.len() == 0)
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
    cand = cand.filter(~_has_class_keyword(pl.col('candidate')) | pl.col('candidate').str.contains(CODE_REGEX))

    # #7: regimen suppression (candidate equals a known regimen token)
    regimen_keys = regimen_index.select(pl.col('regimen_norm').alias('candidate')).unique()
    cand = cand.join(regimen_keys.with_columns(_is_regimen=pl.lit(True)), on='candidate', how='left')
    cand = cand.filter(pl.col('_is_regimen').is_null()).drop('_is_regimen')

    # #11: plural suppression (singular already on M)
    length = pl.col('candidate').str.len_chars()
    cand = cand.with_columns(
        singular=pl.when(pl.col('candidate').str.ends_with('ies'))
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
        entries: See :func:`_parse_aact_entries`.

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
        cleaned.group_by('id', 'candidate')
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
        ``tradeNames`` is untouched -- AACT labels never become trade names, matching
        the pyspark reference.
    """
    aact_grouped = aact_df.group_by('id').agg(pl.col('label').drop_nulls().unique().alias('aact_labels'))

    merged = mol_combined.join(aact_grouped, on='id', how='left').with_columns(
        synonyms_filled=pl.col('synonyms').fill_null([]),
        aact_labels_filled=pl.col('aact_labels').fill_null([]),
    )

    existing_lc = merged.select(
        'id',
        pl.col('synonyms_filled').list.eval(pl.element().struct.field('label').str.to_lowercase()).alias(
            'existing_lc'
        ),
    )

    fresh = (
        merged.select('id', 'aact_labels_filled')
        .explode('aact_labels_filled')
        .drop_nulls('aact_labels_filled')
        .join(existing_lc, on='id', how='left')
        .filter(~pl.col('existing_lc').list.contains(pl.col('aact_labels_filled').str.to_lowercase()))
        .select(
            'id',
            pl.struct(
                pl.col('aact_labels_filled').alias('label'), pl.lit(AACT_SOURCE).alias('source')
            ).alias('new_struct'),
        )
        .group_by('id')
        .agg(pl.col('new_struct'))
    )

    return (
        merged.join(fresh, on='id', how='left')
        .with_columns(
            # array_union already dedups identical structs; list.unique here matches that.
            synonyms=pl.concat_list(pl.col('synonyms_filled'), pl.col('new_struct').fill_null([]))
            .list.unique(maintain_order=True)
            .list.sort()
        )
        .drop('aact_labels', 'synonyms_filled', 'aact_labels_filled', 'new_struct')
    )
