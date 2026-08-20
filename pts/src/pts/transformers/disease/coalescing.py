"""Rules that coalesce duplicate terms before the disease index is built.

Every rule here works on the raw ontology node table and expresses its result the
same way: the term that loses is marked ``meta.deprecated`` and given an
IAO_0100001 entry naming the term that survives.  The index build in
``disease`` then needs no special-casing -- its ``n_clean`` filter drops the
losers and the obsolete-term rollup picks the mapping up from there.
"""

from collections import defaultdict

import polars as pl
from loguru import logger

_IAO_REPLACED_BY = 'http://purl.obolibrary.org/obo/IAO_0100001'
_BPV_DTYPE = pl.List(pl.Struct({'pred': pl.String(), 'val': pl.String()}))
_ONTOLOGY_WEIGHTS = pl.DataFrame(
    [('efo', 1), ('mondo', 2), ('oba', 3), ('orphanet', 4), ('hp', 100)],
    schema=['prefix', 'prefix_rank'],
    orient='row',
)


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
    # --- Step 1: identify active nodes and detect name collisions -----------
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

    if superseded_map.is_empty():
        return n

    # --- Step 2: build the new IAO basicPropertyValues entries -------------
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

    # --- Step 3: unnest meta, apply updates, repack ------------------------
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
    because ``annotate_name_duplicates`` superseded the target afterwards.
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
