from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.ontology import edge, node, schema


def disease_hpo(source: Path, destination: Path) -> None:
    # load the ontology
    logger.debug('loading hpo ontology')
    initial = pl.read_json(source, schema=schema)

    # prepare dataframes
    n = pl.DataFrame(
        initial['graphs'][0][0]['nodes'],
        schema=node,
        strict=False,
    )

    # get parents for each term
    parents = (
        pl.DataFrame(initial['graphs'][0][0]['edges'], schema=edge)
        .filter(pl.col('pred') == 'is_a')
        .with_columns(
            id=pl.col('sub').str.split('/').list.last(),
            parent=pl.col('obj').str.split('/').list.last(),
        )
        .group_by('id')
        .agg(pl.col('parent').alias('parents'))
    )

    # get obsolete terms by getting deprecated nodes
    obsolete_terms = (
        n.unnest('meta')
        .explode('basicPropertyValues')
        .unnest('basicPropertyValues')
        .filter(pl.col('pred') == 'http://www.geneontology.org/formats/oboInOwl#hasAlternativeId')
        .with_columns(
            id=pl.col('id').str.split('/').list.last(),
            obsoleteTerms=pl.col('val').str.split('/').list.last().str.replace(':', '_'),
        )
        .group_by('id')
        .agg(pl.col('obsoleteTerms'))
    )

    # clean the nodes
    n_clean = (
        n.filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        .unnest('meta')
        .with_columns(
            id=pl.col('id').str.split('/').list.last(),
            name=pl.col('lbl'),
            description=pl.col('definition').struct['val'],
            dbXRefs=pl.col('xrefs').list.eval(pl.element().struct.field('val').unique()),
        )
        .drop(
            'basicPropertyValues',
            'comments',
            'definition',
            'deprecated',
            'lbl',
            'subsets',
            'synonyms',
            'type',
            'xrefs',
        )
    )

    n_complete = n_clean.join(
        parents,
        on='id',
        how='left',
    ).join(
        obsolete_terms,
        on='id',
        how='left',
    )

    logger.debug('writing processed hpo data')
    n_complete.write_parquet(destination, compression='gzip')
