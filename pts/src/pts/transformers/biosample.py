"""Build the biosample index from Cell Ontology, Uberon and EFO.

Polars port of gentropy's `BiosampleIndexStep` (`gentropy/biosample_index.py` +
`gentropy/datasource/biosample_ontologies/utils.py` + `gentropy/dataset/biosample_index.py`).

Merges three OBO-Graph-JSON ontology exports into one biosample index, reusing the same
OBO-Graph-JSON parsing pattern as `pts.transformers.disease.disease` (`pts.schemas.ontology.node`
and an iterative Polars join for the transitive ancestor closure) rather than gentropy's own
PySpark implementation, which is a broadcast-UDF graph traversal with no direct Polars equivalent.

Two things gentropy's biosample step does differently from `disease.py`'s EFO handling, kept
faithful here even though they look inconsistent side by side:

* URL-stripping only removes the two specific prefixes gentropy's own regex targets
  (`http://purl.obolibrary.org/obo/`, `http://www.ebi.ac.uk/efo/`), NOT every URL down to its last
  `/`-segment the way `disease.py` does (`.str.split('/').list.last()`). Verified against a real
  production `output/biosample`: cross-references to external ontologies (`ncbigene`, `ensembl`,
  `brain-bican`) are left as full, unstripped URLs -- `disease.py`'s simpler approach would
  over-strip these and produce a different id space than the one actually in production.
* `is_a` AND `part_of` (`BFO_0000050`) edges both count as "parent" relations for biosample's one
  hierarchy (`disease.py` keeps them separate: `is_a` -> `parents`, `BFO_0000050` -> a distinct
  `locationIds` field it doesn't use for closure).
* No `type == 'CLASS'` or `deprecated` filtering -- every node in the graph becomes a biosample
  row. Real ontology exports pull in many non-biosample terms this way (BFO/RO relation terms, GO
  terms referenced for annotation) -- confirmed against a real production `output/biosample`, not
  a bug introduced here.
"""

from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.biosample import BiosampleIndexSchema
from pts.schemas.ontology import edge, node
from pts.transformers.utils.dataset import write_dataset

#: Edge predicates gentropy treats as parent-of relations for biosample (`utils.py`'s `df_parents`
#: filter: `predicate == "is_a" or predicate == "BFO_0000050"`).
_PARENT_PREDICATES = ('is_a', 'BFO_0000050')

#: EFO terms are kept only if they descend from Cell Ontology's root, matching gentropy's
#: `retain_rows_with_ancestor_id(["CL_0000000"])`.
_EFO_ANCESTOR_FILTER = 'CL_0000000'

_LIST_COLUMNS = ('xrefs', 'synonyms', 'parents', 'ancestors', 'descendants', 'children')
_SCALAR_COLUMNS = ('biosampleName', 'description')
_EMPTY_STRING_LIST = pl.Series([[]], dtype=pl.List(pl.String))

#: The exact two URL prefixes gentropy's `utils.py::extract_ontology_from_json` strips (its own
#: `urls_to_remove` list), stripped wherever they occur, not just as a string prefix.
_URL_PREFIXES_TO_STRIP = r'http://purl\.obolibrary\.org/obo/|http://www\.ebi\.ac\.uk/efo/'


def _strip_url_prefixes(expr: pl.Expr) -> pl.Expr:
    """Strip the two ontology URL prefixes gentropy's own code strips, wherever they occur."""
    return expr.str.replace_all(_URL_PREFIXES_TO_STRIP, '')


def biosample(
    source: dict[str, Path],
    destination: Path,
    settings: dict[str, Any],
    _config: Config,
) -> None:
    """Build, merge and write the biosample index from Cell Ontology, Uberon and EFO.

    Args:
        source: `cell_ontology`, `uberon`, `efo` -- OBO-Graph-JSON ontology export paths.
        destination: output biosample index path.
        settings: unused.
        _config: otter Config object, unused.
    """
    logger.info('parsing cell ontology')
    cell_ontology_index = _extract_ontology_index(source['cell_ontology'])
    logger.info('parsing uberon')
    uberon_index = _extract_ontology_index(source['uberon'])
    logger.info('parsing efo')
    efo_index = _extract_ontology_index(source['efo']).filter(pl.col('ancestors').list.contains(_EFO_ANCESTOR_FILTER))

    logger.info('merging ontology indices')
    # Order matters for scalar columns' first-non-null pick: gentropy's own
    # `cell_ontology_index.merge_indices([uberon_index, efo_index])` unions `[uberon, efo, self]`,
    # i.e. cell ontology LAST -- not the more obvious cell/uberon/efo reading order.
    merged = _merge_indices([uberon_index, efo_index, cell_ontology_index])

    logger.info(f'writing {merged.height} biosample records')
    write_dataset(merged, str(destination), schema=BiosampleIndexSchema)


def _extract_ontology_index(path: Path) -> pl.DataFrame:
    """Parse one OBO-Graph-JSON ontology export into the biosample row shape.

    Args:
        path: OBO-Graph-JSON file path.

    Returns:
        One row per graph node: `biosampleId`, `biosampleName`, `description`, `xrefs`,
        `synonyms`, `parents`, `children`, `ancestors`, `descendants`.
    """
    handle = StorageHandle(str(path))
    raw = pl.read_json(handle.open())  # ty:ignore[invalid-argument-type]
    graph = raw['graphs'][0][0]

    nodes = pl.DataFrame(graph['nodes'], schema=node).unnest('meta')
    edges = pl.DataFrame(graph['edges'], schema=edge)

    flat_nodes = nodes.select(
        _strip_url_prefixes(pl.col('id')).alias('biosampleId'),
        pl.coalesce('lbl', 'id').alias('biosampleName'),
        pl.col('definition').struct.field('val').alias('description'),
        pl.col('xrefs').list.eval(pl.element().struct.field('val')).fill_null(_EMPTY_STRING_LIST).alias('xrefs'),
        pl
        .col('synonyms')
        .list.eval(pl.element().struct.field('val'))
        .fill_null(_EMPTY_STRING_LIST)
        .alias('synonyms'),
    )

    parent_edges = (
        edges
        .select(
            _strip_url_prefixes(pl.col('sub')).alias('biosampleId'),
            _strip_url_prefixes(pl.col('pred')).alias('predicate'),
            _strip_url_prefixes(pl.col('obj')).alias('parent'),
        )
        .filter(pl.col('predicate').is_in(_PARENT_PREDICATES))
    )

    parents = parent_edges.group_by('biosampleId').agg(pl.col('parent').unique().alias('parents'))
    children = (
        parent_edges
        .group_by('parent')
        .agg(pl.col('biosampleId').unique().alias('children'))
        .rename({'parent': 'biosampleId'})
    )

    with_relations = flat_nodes.join(parents, on='biosampleId', how='left').join(children, on='biosampleId', how='left')

    ancestors, descendants = _ancestor_and_descendant_closure(with_relations.select('biosampleId', 'parents'))

    return with_relations.join(ancestors, on='biosampleId', how='left').join(descendants, on='biosampleId', how='left')


def _ancestor_and_descendant_closure(direct: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Compute the full transitive ancestor closure (and its inverse, descendants).

    Iteratively joins the current relationship frontier against `parents` until no new
    relationships are found -- the same fixed-point idiom `pts.transformers.disease.disease` uses
    for EFO's ancestor closure, since Polars has no native graph-traversal primitive.

    Args:
        direct: `biosampleId`, `parents` (nullable `list[str]`) -- one row per node.

    Returns:
        `(ancestors, descendants)`, each a `biosampleId` + `list[str]` column, with one row per
        node that has at least one (a node with none is absent, not null-valued).
    """
    frontier = direct.filter(pl.col('parents').is_not_null()).explode('parents').rename({'parents': 'ancestor'})
    all_relationships = frontier

    while frontier.height > 0:
        frontier = (
            frontier
            .join(direct, left_on='ancestor', right_on='biosampleId')
            .filter(pl.col('parents').is_not_null())
            .explode('parents')
            .select(pl.col('biosampleId'), pl.col('parents').alias('ancestor'))
        )
        if frontier.height == 0:
            break
        all_relationships = pl.concat([all_relationships, frontier])

    ancestors = all_relationships.group_by('biosampleId').agg(pl.col('ancestor').unique().alias('ancestors'))
    descendants = (
        all_relationships
        .select(pl.col('ancestor').alias('biosampleId'), pl.col('biosampleId').alias('descendant'))
        .group_by('biosampleId')
        .agg(pl.col('descendant').unique().alias('descendants'))
    )
    return ancestors, descendants


def _merge_indices(indices: list[pl.DataFrame]) -> pl.DataFrame:
    """Merge several ontology indices into one, deduplicating on `biosampleId`.

    Polars port of `BiosampleIndex.merge_indices`: scalar columns take the first non-null value
    across sources; list columns take the flattened, deduplicated union -- which also normalises
    every list column to `[]` (never null) for any id absent from all but null-valued sources.

    Args:
        indices: ontology indices to merge, as produced by `_extract_ontology_index`.

    Returns:
        One row per distinct `biosampleId`.
    """
    return (
        pl.concat(indices, how='vertical')
        .group_by('biosampleId')
        .agg(
            *(pl.col(c).drop_nulls().first().alias(c) for c in _SCALAR_COLUMNS),
            *(
                pl.col(c).list.explode(keep_nulls=False, empty_as_null=False).drop_nulls().unique().alias(c)
                for c in _LIST_COLUMNS
            ),
        )
    )
