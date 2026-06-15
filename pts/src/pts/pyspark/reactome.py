"""Reactome pathway graph dataset generation.

Ported from Reactome.scala in platform-etl-backend.
Builds a directed acyclic graph of Reactome human pathways and computes
ancestor/descendant/children/parents/path relationships for each node.
"""

from __future__ import annotations

from typing import Any

import networkx as nx
from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


def _clean_pathways(df: DataFrame) -> DataFrame:
    """Filter for human pathways and rename columns to id, name."""
    return df.filter(f.col('_c2') == 'Homo sapiens').drop('_c2').toDF('id', 'name')


def _build_graph_documents(
    spark: SparkSession,
    vertices: DataFrame,
    edges: DataFrame,
) -> DataFrame:
    """Build graph ancestry documents from vertices and edges DataFrames.

    Args:
        spark: Active SparkSession.
        vertices: DataFrame with columns [id, name].
        edges: DataFrame with columns [src, dst].

    Returns:
        DataFrame with columns [id, label, ancestors, descendants, children, parents, path].
    """
    v_list = [(r['id'], r['name']) for r in vertices.collect()]
    e_list = [(r['src'], r['dst']) for r in edges.collect()]

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
        ancestors = list(nx.ancestors(g, node_id))
        descendants = list(nx.descendants(g, node_id))
        children = list(g.successors(node_id))
        parents = list(g.predecessors(node_id))
        paths = []
        for root in roots:
            if nx.has_path(g, root, node_id):
                paths.extend(list(p) for p in nx.all_simple_paths(g, root, node_id))
        # Sort paths deterministically: shortest first, then by root node ID.
        # This ensures the first path (used for topLevelTerm in the target step)
        # is consistent across runs.
        paths.sort(key=lambda p: (len(p), p[0]))
        rows.append((node_id, label, ancestors, descendants, children, parents, paths))

    return spark.createDataFrame(
        rows,
        schema='id STRING, label STRING, ancestors ARRAY<STRING>, descendants ARRAY<STRING>, '
        'children ARRAY<STRING>, parents ARRAY<STRING>, path ARRAY<ARRAY<STRING>>',
    )


def reactome(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate Reactome pathway graph dataset."""
    spark = Session(app_name='reactome', properties=properties).spark

    logger.info('Reading Reactome pathway inputs')
    pathways = spark.read.option('sep', '\t').option('header', 'false').csv(source['pathways'])
    relations = spark.read.option('sep', '\t').option('header', 'false').csv(source['relations'])

    edges = relations.toDF('src', 'dst')
    clean = _clean_pathways(pathways)

    logger.info('Building Reactome graph')
    result = _build_graph_documents(spark, clean, edges).distinct()

    logger.info(f'Writing Reactome output to {destination}')
    result.write.mode('overwrite').parquet(destination)
