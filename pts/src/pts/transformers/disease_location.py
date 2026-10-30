"""Build the disease location dataset from MONDO's disease-has-location axioms."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas import ontology
from pts.schemas.disease_location import disease_location_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset

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


def most_specific_locations(pairs: pl.DataFrame, biosample: pl.DataFrame) -> pl.DataFrame:
    """Keep the locations the biosample index knows, and of those only the most specific.

    A location is dropped when it is an ancestor, in the biosample index, of another location
    on the same disease.

    Args:
        pairs: ``diseaseId`` and ``biosampleId`` pairs.
        biosample: biosample index rows, with ``biosampleId`` and ``ancestors``.

    Returns:
        The surviving ``diseaseId`` and ``biosampleId`` pairs.
    """
    known = pairs.join(biosample.select('biosampleId'), on='biosampleId', how='semi')
    broader = (
        known
        .join(biosample.select('biosampleId', 'ancestors'), on='biosampleId')
        .explode('ancestors')
        .drop_nulls('ancestors')
        .select('diseaseId', biosampleId=pl.col('ancestors'))
        .unique()
    )
    return known.join(broader, on=['diseaseId', 'biosampleId'], how='anti')


def disease_location(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Build the disease location dataset.

    Args:
        source: ``mondo`` (OBO-Graph JSON), ``disease`` and ``biosample`` dataset paths.
        destination: output dataset path.
        settings: unused.
        config: unused.
    """
    logger.debug('loading mondo')
    f = StorageHandle(source['mondo']).open()
    graph = pl.read_json(f, schema=ontology.schema)['graphs'][0][0]  # ty:ignore[invalid-argument-type]
    edges = pl.DataFrame(graph['edges'], schema=ontology.edge)
    axioms = pl.DataFrame(graph['logicalDefinitionAxioms'])

    disease = scan_dataset(source['disease']).select('id', 'ancestors').collect()
    biosample = scan_dataset(source['biosample']).select('biosampleId', 'ancestors').collect()

    locations = (
        most_specific_locations(inherit_locations(stated_locations(edges, axioms), disease), biosample)
        .sort('diseaseId', 'biosampleId')
        .select(disease_location_schema.names())
    )
    logger.info(f'{locations.height} locations on {locations["diseaseId"].n_unique()} diseases')
    write_dataset(locations, destination)
