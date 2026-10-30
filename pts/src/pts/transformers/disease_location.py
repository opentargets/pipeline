"""Build the disease location dataset from MONDO's disease-has-location axioms."""

import polars as pl

LOCATED_IN = 'http://purl.obolibrary.org/obo/RO_0004026'


def _short_id(expr: pl.Expr) -> pl.Expr:
    return expr.str.split('/').list.last()


def stated_locations(edges: pl.DataFrame, axioms: pl.DataFrame) -> pl.DataFrame:
    """Collect every location MONDO states for a class, as edges or inside class definitions.

    Args:
        edges: OBO-Graph edges, with ``sub``, ``pred`` and ``obj``.
        axioms: OBO-Graph logical definition axioms, with ``definedClassId`` and ``restrictions``.

    Returns:
        Distinct ``diseaseId`` and ``biosampleId`` pairs, as short ids.
    """
    from_edges = edges.filter(pl.col('pred') == LOCATED_IN).select(
        diseaseId=_short_id(pl.col('sub')),
        biosampleId=_short_id(pl.col('obj')),
    )
    from_axioms = (
        axioms
        .select('definedClassId', 'restrictions')
        .explode('restrictions', empty_as_null=True)
        .drop_nulls('restrictions')
        .filter(pl.col('restrictions').struct['propertyId'] == LOCATED_IN)
        .select(
            diseaseId=_short_id(pl.col('definedClassId')),
            biosampleId=_short_id(pl.col('restrictions').struct['fillerId']),
        )
    )
    return pl.concat([from_edges, from_axioms]).drop_nulls().unique()


def inherit_locations(stated: pl.DataFrame, disease: pl.DataFrame) -> pl.DataFrame:
    """Give every disease the stated locations of itself and of each of its ancestors.

    Args:
        stated: ``diseaseId`` and ``biosampleId`` pairs, as stated by the ontology.
        disease: disease index rows, with ``id`` and ``ancestors``.

    Returns:
        Distinct ``diseaseId`` and ``biosampleId`` pairs, one disease per index row.
    """
    return (
        disease
        .select(
            diseaseId=pl.col('id'),
            source=pl.concat_list(pl.col('id'), pl.col('ancestors').fill_null([])),
        )
        .explode('source')
        .join(stated.rename({'diseaseId': 'source'}), on='source')
        .select('diseaseId', 'biosampleId')
        .unique()
    )
