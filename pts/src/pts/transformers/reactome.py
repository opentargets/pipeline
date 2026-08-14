"""Reactome pathway graph dataset generation.

Ported from Reactome.scala in platform-etl-backend.
Builds a directed acyclic graph of Reactome human pathways and computes
ancestor/descendant/children/parents/path relationships for each node.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import networkx as nx
import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.reactome import reactome_schema

HUMAN = 'Homo sapiens'


def _read_tsv(path: str | Path, columns: list[str]) -> pl.DataFrame:
    """Read a headerless Reactome TSV.

    Reactome ships plain tab separated values with no quoting: pathway names are
    free text and a double quote in one would be a literal character, not a
    delimiter. `quote_char=None` keeps it that way.
    """
    h = StorageHandle(path)
    return pl.read_csv(
        h.open(),
        separator='\t',
        has_header=False,
        quote_char=None,
        new_columns=columns,
        schema_overrides=dict.fromkeys(columns, pl.String),
    )


def _clean_pathways(df: pl.DataFrame) -> pl.DataFrame:
    """Filter for human pathways and rename columns to id, name."""
    return df.filter(pl.col(df.columns[2]) == HUMAN).select(
        pl.col(df.columns[0]).alias('id'),
        pl.col(df.columns[1]).alias('name'),
    )


def _build_graph_documents(vertices: pl.DataFrame, edges: pl.DataFrame) -> pl.DataFrame:
    """Build graph ancestry documents from vertices and edges DataFrames.

    Only `vertices` become output rows. Edges may reference pathways of other
    species; those nodes join the graph as traversal context but are never
    emitted. Reactome relations do not cross species, so they contribute nothing
    to a human node's ancestry in practice.

    Args:
        vertices: DataFrame with columns [id, name].
        edges: DataFrame with columns [src, dst].

    Returns:
        DataFrame with columns [id, label, ancestors, descendants, children, parents, path].
    """
    v_list = list(zip(vertices['id'].to_list(), vertices['name'].to_list(), strict=True))
    e_list = list(zip(edges['src'].to_list(), edges['dst'].to_list(), strict=True))

    g: nx.DiGraph = nx.DiGraph()
    g.add_nodes_from(v for v, _ in v_list)
    for src, dst in e_list:
        g.add_edge(src, dst)

    # Guard against cycles (the Scala version used DirectedAcyclicGraph which enforced acyclicity)
    if not nx.is_directed_acyclic_graph(g):
        cycles = list(nx.simple_cycles(g))
        logger.warning(f'Input graph contains {len(cycles)} cycle(s); removing back-edges')
        for cycle in cycles:
            g.remove_edge(cycle[-1], cycle[0])

    roots = [n for n in g.nodes if g.in_degree(n) == 0]

    rows = []
    for node_id, label in v_list:
        # nx.ancestors and nx.descendants return sets, and successors/predecessors
        # follow edge insertion order. Sorting all four keeps the output identical
        # between runs on identical input; without it the string hash seed alone
        # changes the element order of every ancestors and descendants list.
        ancestors = sorted(nx.ancestors(g, node_id))
        descendants = sorted(nx.descendants(g, node_id))
        children = sorted(g.successors(node_id))
        parents = sorted(g.predecessors(node_id))
        paths = []
        for root in roots:
            if nx.has_path(g, root, node_id):
                paths.extend(list(p) for p in nx.all_simple_paths(g, root, node_id))
        # Sort paths deterministically: shortest first, then by root node ID.
        # This ensures the first path (used for topLevelTerm in the target step)
        # is consistent across runs.
        paths.sort(key=lambda p: (len(p), p[0]))
        rows.append({
            'id': node_id,
            'label': label,
            'ancestors': ancestors,
            'descendants': descendants,
            'children': children,
            'parents': parents,
            'path': paths,
        })

    return pl.DataFrame(rows, schema=reactome_schema)


def reactome(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate Reactome pathway graph dataset."""
    logger.info('Reading Reactome pathway inputs')
    pathways = _read_tsv(source['pathways'], ['id', 'name', 'species'])
    relations = _read_tsv(source['relations'], ['src', 'dst'])

    clean = _clean_pathways(pathways)
    logger.info(f'Found {clean.height} {HUMAN} pathways in {relations.height} relations')

    logger.info('Building Reactome graph')
    result = _build_graph_documents(clean, relations)

    logger.info(f'Writing Reactome output to {destination}')
    result.write_parquet(destination)
