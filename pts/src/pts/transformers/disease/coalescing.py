"""Rules that coalesce duplicate terms before the disease index is built.

Every rule here works on the raw ontology node table and expresses its result the
same way: the term that loses is marked ``meta.deprecated`` and given an
IAO_0100001 entry naming the term that survives.  The index build in
``disease`` then needs no special-casing -- its ``n_clean`` filter drops the
losers and the obsolete-term rollup picks the mapping up from there.
"""

import itertools
from collections import defaultdict

import polars as pl
from loguru import logger

from pts.transformers.disease.ontology_ids import (
    FINE_NAMESPACES,
    ONTOLOGY_NAMESPACES,
    name_bags,
    normalised_xrefs,
)

_IAO_REPLACED_BY = 'http://purl.obolibrary.org/obo/IAO_0100001'
_BPV_DTYPE = pl.List(pl.Struct({'pred': pl.String(), 'val': pl.String()}))
_ONTOLOGY_WEIGHTS = pl.DataFrame(
    [('efo', 1), ('mondo', 2), ('oba', 3), ('orphanet', 4), ('hp', 100)],
    schema=['prefix', 'prefix_rank'],
    orient='row',
)

# Only disease and phenotype ontologies take part in cross-reference merging.
# GO and MP are deliberately absent: a biological process and a mouse phenotype
# are not a duplicate of a disease even when they share a reference and a name.
_MERGE_PREFIXES = frozenset({'efo', 'mondo', 'hp', 'orphanet', 'oba'})

# A cross-reference shared by more terms than this is a hub, not an identity.
_MAX_XREF_FANOUT = 40


def annotate_name_duplicates(n: pl.DataFrame) -> pl.DataFrame:
    """Annotate name-collision nodes in the raw ontology node table.

    Finds non-deprecated CLASS nodes whose labels are identical when compared
    case-insensitively (e.g. 'Acidosis' vs 'acidosis').  For each collision
    group the node from the lower-priority ontology is marked as superseded:
      - meta.deprecated is set to True
      - An IAO_0100001 basicPropertyValues entry is added pointing to the
        canonical (higher-priority) node's full URL.

    Ontology priority (ascending, lower rank wins): efo < mondo < oba <
    orphanet < hp < other.

    This allows the standard n_clean filter (~deprecated) and the existing
    obsolete_ids / replace_obsolete_terms pipeline to handle the resolution
    transparently without any additional special-casing.

    Args:
        n: Raw node DataFrame with the ``node`` schema (id, lbl, meta, type).

    Returns:
        DataFrame with the same shape and schema as ``n``, with meta updated
        for superseded nodes.
    """
    # identify active nodes and detect name collisions
    active = n.filter(
        pl.col('type') == 'CLASS',
        ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
    ).with_columns(
        name_lower=pl.col('lbl').str.to_lowercase(),
        prefix=pl.col('id').str.split('/').list.last().str.split('_').list.first().str.to_lowercase(),
    )

    collision_ids = (
        active
        .filter(pl.col('name_lower').is_duplicated())
        .join(_ONTOLOGY_WEIGHTS, on='prefix', how='left')
        .with_columns(pl.col('prefix_rank').fill_null(99))
        .sort(['name_lower', 'prefix_rank'])
        .with_columns(row_rank=pl.int_range(pl.len()).over('name_lower'))
    )

    canonical = collision_ids.filter(pl.col('row_rank') == 0).select(
        pl.col('name_lower'), pl.col('id').alias('canonical_url')
    )

    superseded_map = (
        collision_ids
        .filter(pl.col('row_rank') > 0)
        .join(canonical, on='name_lower')
        .select(
            pl.col('id').alias('superseded_url'),
            pl.col('canonical_url'),
        )
    )

    return _supersede(n, superseded_map)


def _supersede(n: pl.DataFrame, superseded_map: pl.DataFrame) -> pl.DataFrame:
    """Mark superseded nodes deprecated and point them at their canonical node.

    Shared by every coalescing rule so they all hand the same thing to the
    downstream ``n_clean`` filter and ``obsolete_ids`` extraction: the losing
    node gets ``meta.deprecated = True`` and an IAO_0100001 entry naming the
    surviving node's full URL.

    Args:
        n: Raw node DataFrame with the ``node`` schema.
        superseded_map: Two columns, ``superseded_url`` and ``canonical_url``.

    Returns:
        DataFrame with the same shape and schema as ``n``.
    """
    if superseded_map.is_empty():
        return n

    # build the new IAO basicPropertyValues entries
    iao_additions = (
        superseded_map
        .select(
            pl.col('superseded_url').alias('id'),
            pl.struct(
                pred=pl.lit(_IAO_REPLACED_BY),
                val=pl.col('canonical_url'),
            ).alias('iao_entry'),
        )
        .group_by('id')
        .agg(pl.col('iao_entry').alias('iao_entries'))
    )

    # unnest meta, apply the updates, repack
    n_unnested = n.unnest('meta').join(iao_additions, on='id', how='left')

    return (
        n_unnested
        .with_columns(
            deprecated=pl.when(pl.col('iao_entries').is_not_null()).then(True).otherwise(pl.col('deprecated')),
            basicPropertyValues=pl
            .when(pl.col('iao_entries').is_not_null())
            .then(
                pl
                .col('basicPropertyValues')
                .fill_null(pl.Series([[]], dtype=_BPV_DTYPE))
                .list.concat(pl.col('iao_entries'))
            )
            .otherwise(pl.col('basicPropertyValues')),
        )
        .drop('iao_entries')
        .with_columns(
            meta=pl.struct(
                basicPropertyValues=pl.col('basicPropertyValues'),
                comments=pl.col('comments'),
                definition=pl.col('definition'),
                deprecated=pl.col('deprecated'),
                subsets=pl.col('subsets'),
                synonyms=pl.col('synonyms'),
                xrefs=pl.col('xrefs'),
            )
        )
        .drop(
            'basicPropertyValues',
            'comments',
            'definition',
            'deprecated',
            'subsets',
            'synonyms',
            'xrefs',
        )
        .select(n.columns)
    )


def resolve_replacement_chains(n: pl.DataFrame) -> pl.DataFrame:
    """Point every replacement pointer at a term that survives ``n_clean``.

    An IAO_0100001 entry may name a node that is itself deprecated, either
    because the source ontology obsoleted a term into another obsolete term or
    because one of the coalescing rules superseded the target afterwards.
    ``remap_edges`` and ``obsolete_ids`` both resolve a single hop, so the far
    end of such a chain is lost: the intermediate node is dropped by
    ``n_clean``, taking with it the mapping from the original term, and
    evidence carrying that term no longer reaches the surviving one.

    Each pointer is followed to the first term that survives ``n_clean`` -- a
    CLASS row that is not deprecated -- and is rewritten only when such a term
    is actually reached.  A hop is only taken when it is unambiguous: a
    deprecated node naming two *different* replacements is a curation decision
    to leave alone, as are cycles and self-references, which name no live term
    and would only rotate the pointer.

    Two other endings are left as they stand, because neither leads anywhere
    ``n_clean`` keeps.  A chain can run into a term the source ontology
    obsoleted without naming a successor, and it can run into an id the graph
    does not carry at all -- some entries name a CURIE where every node id is a
    URL.  Both are properties of the source data rather than something this can
    repair.

    Args:
        n: Node DataFrame whose IAO_0100001 entries may form chains.

    Returns:
        DataFrame with the same shape and schema as ``n``.
    """
    pointers = (
        n
        .unnest('meta')
        .explode('basicPropertyValues')
        .unnest('basicPropertyValues')
        .filter(pl.col('deprecated'), pl.col('pred') == _IAO_REPLACED_BY)
        .select('id', 'val')
    )
    if pointers.is_empty():
        return n

    # A chain has only landed somewhere useful if the term it reaches survives
    # n_clean, which keeps CLASS rows that are not deprecated.  Being absent
    # from the graph is not the same as being live: some entries name a CURIE
    # where every node id is a URL, and those match no term at all.
    live = set(
        n
        .filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        .get_column('id')
        .to_list()
    )
    known = set(n.get_column('id').to_list())

    targets: dict[str, set[str]] = defaultdict(set)
    for source, target in pointers.iter_rows():
        targets[source].add(target)
    # Only an unambiguous pointer can be followed through.  Ambiguity means two
    # *different* replacements, so a term repeating the same pointer still names
    # a single successor.
    ambiguous = {source: found for source, found in targets.items() if len(found) > 1}
    onward = {source: next(iter(found)) for source, found in targets.items() if source not in ambiguous}

    # Terms naming several replacements stop any chain that reaches them, so
    # name them for the curator rather than dropping them in silence.
    if ambiguous:
        logger.info(f'{len(ambiguous)} obsolete terms name more than one replacement, so chains through them stop')
        for source, found in sorted(ambiguous.items()):
            logger.debug(f'ambiguous replacement: {source} -> {", ".join(sorted(found))}')

    resolved: dict[str, str] = {}
    cycles: dict[frozenset[str], str] = {}
    for start in sorted({target for found in targets.values() for target in found}):
        seen = {start}
        path = [start]
        current = start
        cycled = False
        while current in onward:
            nxt = onward[current]
            if nxt in seen:
                # A cycle or self-reference names no live term, so following it
                # would only rotate the pointer.  Leave it as it stands.
                cycled = True
                loop = path[path.index(nxt):]
                cycles.setdefault(frozenset(loop), ' -> '.join([*loop, nxt]))
                break
            seen.add(nxt)
            path.append(nxt)
            current = nxt
        if not cycled and current != start and current in live:
            resolved[start] = current

    # A cycle is a curation error upstream, so name it rather than leaving an
    # unexplained gap.  Every term on a cycle reaches it, so report it once.
    if cycles:
        logger.info(f'{len(cycles)} replacement chains close a cycle and are left as they stand')
        for cycle in sorted(cycles.values()):
            logger.debug(f'cyclic replacement: {cycle}')

    # The two ways a pointer fails call for different follow-ups, so report
    # them apart: naming a term the graph carries but n_clean drops is ours to
    # chase, naming an id the graph lacks belongs to the source ontology.  A
    # term naming itself is counted with the cycles rather than here.
    landing = [
        resolved.get(target, target)
        for source, found in targets.items()
        for target in found
        if target != source
    ]
    dangling = sum(1 for target in landing if target in known and target not in live)
    unknown = sum(1 for target in landing if target not in known)
    if dangling:
        logger.info(f'{dangling} replacement pointers still name a term that will be dropped')
    if unknown:
        logger.info(f'{unknown} replacement pointers name an id the ontology does not carry')

    if not resolved:
        return n

    logger.debug(f'resolved {len(resolved)} replacement pointers through a chain')
    return n.with_columns(
        meta=pl.col('meta').struct.with_fields(
            basicPropertyValues=pl
            .col('meta')
            .struct.field('basicPropertyValues')
            .list.eval(
                pl.struct(
                    pred=pl.element().struct.field('pred'),
                    val=pl
                    .when(pl.element().struct.field('pred') == _IAO_REPLACED_BY)
                    .then(pl.element().struct.field('val').replace(resolved))
                    .otherwise(pl.element().struct.field('val')),
                )
            ),
        )
    )


def remap_edges(e: pl.DataFrame, n: pl.DataFrame) -> pl.DataFrame:
    """Replace deprecated node URLs in edges with their canonical replacements.

    Extracts the deprecated→canonical mapping from IAO_0100001 basicPropertyValues
    entries in ``n``, then rewrites any ``sub`` or ``obj`` in ``e`` that references
    a deprecated node.  Self-loops and duplicate edges introduced by the remapping
    are removed.

    Args:
        e: Edge DataFrame with columns ``sub``, ``pred``, ``obj`` (full URLs).
        n: Node DataFrame (node schema), typically after ``annotate_name_duplicates``.

    Returns:
        Remapped edge DataFrame with the same columns as ``e``.
    """
    id_remap = (
        n
        .unnest('meta')
        .explode('basicPropertyValues')
        .unnest('basicPropertyValues')
        .filter(
            pl.col('deprecated'),
            pl.col('pred') == _IAO_REPLACED_BY,
        )
        .select(
            pl.col('id').alias('old_url'),
            pl.col('val').alias('new_url'),
        )
    )

    return (
        e
        .join(
            id_remap.rename({'old_url': 'sub', 'new_url': 'sub_new'}),
            on='sub',
            how='left',
        )
        .join(
            id_remap.rename({'old_url': 'obj', 'new_url': 'obj_new'}),
            on='obj',
            how='left',
        )
        .with_columns(
            sub=pl.coalesce('sub_new', 'sub'),
            obj=pl.coalesce('obj_new', 'obj'),
        )
        .drop('sub_new', 'obj_new')
        .filter(pl.col('sub') != pl.col('obj'))
        .unique()
        .select(e.columns)
    )


def _ancestor_lookup(e: pl.DataFrame, of_interest: set[str]) -> dict[str, set[str]]:
    """Transitive is_a ancestors, computed only for the terms we need.

    The full closure is built later in ``disease``; here we only need it to
    reject candidate pairs standing in a hierarchy, so walking up from the
    candidates is far cheaper than closing the whole graph.
    """
    parents: dict[str, list[str]] = defaultdict(list)
    is_a = (
        e
        .filter(pl.col('pred') == 'is_a')
        .select(
            child=pl.col('sub').str.split('/').list.last(),
            parent=pl.col('obj').str.split('/').list.last(),
        )
    )
    for child, parent in is_a.iter_rows():
        parents[child].append(parent)

    ancestors: dict[str, set[str]] = {}
    for term in sorted(of_interest):
        seen: set[str] = set()
        frontier = list(parents.get(term, []))
        while frontier:
            current = frontier.pop()
            if current in seen:
                continue
            seen.add(current)
            frontier.extend(parents.get(current, []))
        ancestors[term] = seen
    return ancestors


def _merge_groups(edges: list[tuple[str, str]]) -> list[list[str]]:
    """Union-find over candidate edges, returning sorted groups of >1 term.

    Both the union order and the group contents are sorted so the result does
    not depend on set or dict iteration order.
    """
    parent: dict[str, str] = {}

    def find(term: str) -> str:
        parent.setdefault(term, term)
        root = term
        while parent[root] != root:
            root = parent[root]
        while parent[term] != root:  # path compression
            parent[term], term = root, parent[term]
        return root

    for a, b in sorted(edges):
        root_a, root_b = find(a), find(b)
        if root_a != root_b:
            # Union by id keeps the representative deterministic.
            high, low = max(root_a, root_b), min(root_a, root_b)
            parent[high] = low

    grouped: dict[str, list[str]] = defaultdict(list)
    for term in sorted(parent):
        grouped[find(term)].append(term)
    return [sorted(members) for _, members in sorted(grouped.items()) if len(members) > 1]


def annotate_xref_duplicates(n: pl.DataFrame, e: pl.DataFrame) -> pl.DataFrame:
    """Coalesce cross-ontology duplicates that ``annotate_name_duplicates`` misses.

    The identical-label rule only merges terms whose labels match exactly, so
    the same disease carried by two ontologies under different wording stays
    split, and the evidence with it (opentargets/issues#4446).  Two terms are
    treated as the same entity when, **cross-ontology only**, either:

    - one term's dbXRefs names the other term's ontology id; or
    - they share a fine-grained cross-reference *and* an exact name, and
      neither is an is_a ancestor of the other.

    Only the second arm carries the hierarchy guard, deliberately.  A direct
    cross-reference is a curator asserting that two terms are the same thing,
    which outranks an is_a edge between them -- where the two disagree it is
    usually the edge that is wrong, as when an ontology files an eponym under
    the disease it names.  A shared third-party code asserts nothing of the
    kind, and is exactly where granularity mismatches put a parent and child
    on the same code, so there the guard is needed.

    Each guardrail exists because dropping it costs precision: publication and
    protein references are not identity; coarse classification codes are
    many-to-one; a cross-reference carried by more than ``_MAX_XREF_FANOUT``
    terms is a hub rather than an identifier; and a resulting group holding two
    terms of one ontology is dropped whole, since it bridges those two rather
    than equating them.

    The surviving term is the one from the higher-priority ontology
    (``_ONTOLOGY_WEIGHTS``), ties broken by id.  Output matches
    ``annotate_name_duplicates``: losers are marked deprecated with an
    IAO_0100001 pointer, so the existing ``n_clean`` filter and obsolete-term
    machinery resolve them with no further special-casing.

    Args:
        n: Raw node DataFrame with the ``node`` schema.
        e: Edge DataFrame with ``sub``, ``pred``, ``obj`` (full URLs), used for
            the is_a hierarchy check.  Pass edges already remapped through any
            earlier merge: an unresolved merge hides ancestor links, and the
            hierarchy guard then lets a parent and child merge.

    Returns:
        DataFrame with the same shape and schema as ``n``.
    """
    active = (
        n
        .filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        .with_columns(short_id=pl.col('id').str.split('/').list.last())
        .with_columns(prefix=pl.col('short_id').str.split('_').list.first().str.to_lowercase())
        .filter(pl.col('prefix').is_in(_MERGE_PREFIXES))
    )
    if active.height < 2:
        return n

    prefix_of = dict(active.select('short_id', 'prefix').iter_rows())
    url_of = dict(active.select('short_id', 'id').iter_rows())
    xrefs = normalised_xrefs(active)

    # --- arm 1: one term directly cross-references the other ----------------
    direct = (
        xrefs
        .filter(pl.col('namespace').is_in(ONTOLOGY_NAMESPACES))
        .with_columns(
            target=pl
            .when(pl.col('namespace') == 'ORPHANET')
            .then(pl.lit('Orphanet_') + pl.col('code'))
            .otherwise(pl.col('namespace') + pl.lit('_') + pl.col('code')),
        )
        .filter(pl.col('target').is_in(list(prefix_of)))
        .select('short_id', 'target')
        .unique()
    )
    candidates = [
        (a, b)
        for a, b in direct.iter_rows()
        if a != b and prefix_of[a] != prefix_of[b]
    ]

    # --- arm 2: shared fine-grained cross-reference, corroborated by a name --
    shared = (
        xrefs
        .filter(pl.col('namespace').is_in(FINE_NAMESPACES))
        .group_by('canonical')
        .agg(pl.col('short_id').sort().alias('members'))
        .with_columns(fanout=pl.col('members').list.len())
        .filter(pl.col('fanout') > 1, pl.col('fanout') <= _MAX_XREF_FANOUT)
        .sort('canonical')
    )
    shared_pairs = sorted({
        (a, b)
        for members in shared['members'].to_list()
        for a, b in itertools.combinations(members, 2)
        if prefix_of[a] != prefix_of[b]
    })

    if shared_pairs:
        names_of = name_bags(active)
        named = [(a, b) for a, b in shared_pairs if names_of.get(a, set()) & names_of.get(b, set())]
        ancestors_of = _ancestor_lookup(e, {t for pair in named for t in pair})
        candidates += [
            (a, b)
            for a, b in named
            if b not in ancestors_of.get(a, set()) and a not in ancestors_of.get(b, set())
        ]

    if not candidates:
        return n

    # --- gate the groups and choose each survivor ---------------------------
    ranks = dict(_ONTOLOGY_WEIGHTS.iter_rows())
    superseded: list[dict[str, str]] = []
    bridging: list[list[str]] = []
    for members in _merge_groups(candidates):
        prefixes = [prefix_of[term] for term in members]
        if len(set(prefixes)) != len(prefixes):
            # Two terms of one ontology: the group bridges rather than equates.
            bridging.append(members)
            continue
        canonical = min(members, key=lambda term: (ranks.get(prefix_of[term], 99), term))
        superseded += [
            {'superseded_url': url_of[term], 'canonical_url': url_of[canonical]}
            for term in members
            if term != canonical
        ]

    # A group is dropped whole, so one term joining it withdraws the merges of
    # everything already in it.  That can happen from an upstream edit alone, so
    # name the groups rather than letting merges disappear between releases.
    if bridging:
        logger.info(f'{len(bridging)} candidate groups bridge two terms of one ontology and are dropped')
        for members in bridging:
            logger.debug(f'bridging group: {", ".join(members)}')

    logger.debug(f'cross-reference coalescing supersedes {len(superseded)} terms')
    superseded_map = pl.DataFrame(
        superseded,
        schema={'superseded_url': pl.String(), 'canonical_url': pl.String()},
        orient='row',
    )
    return _supersede(n, superseded_map)
