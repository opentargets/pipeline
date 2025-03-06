from pathlib import Path

import polars as pl
from loguru import logger

from pts.schemas.ontology import edge as ontology_edge
from pts.schemas.ontology import node as ontology_node
from pts.schemas.ontology import schema as ontology_schema


def disease_phenotype(source: list[Path], destination: Path) -> None:
    # source definition
    disease_path = source[0]
    phenotype_path = source[1]
    mondo_path = source[2]

    # NOTE: This code is horrible. We have to figure out a way to make this in
    # a better way. Ontologies are very messy. Relationships are all over the
    # place, ids have different forms, links are indirect, etc.
    # A good idea would be to first gather all the crossrefs for each entity from
    # the multiple different places they can be, normalize them, and then perform
    # the join after. Right now it has been down slowly and painfully.
    #
    # Abandon all hope, ye who enter here.

    # load the sources
    logger.debug('loading disease dataset')
    df_disease = pl.read_parquet(disease_path)
    logger.debug('loading phenotype dataset')
    df_phenotype = pl.read_csv(
        phenotype_path,
        separator='\t',
        has_header=True,
        comment_prefix='#',
    )
    logger.debug('loading mondo dataset')
    df_mondo = pl.read_json(mondo_path, schema=ontology_schema)

    ############################################################################
    # mondo dataset
    # prepare the mondo dataframe
    mondo_nodes = pl.DataFrame(
        df_mondo['graphs'][0][0]['nodes'],
        schema=ontology_node,
    )

    # get phenotypes in edges
    mondo_phenotypes = (
        pl.DataFrame(df_mondo['graphs'][0][0]['edges'], schema=ontology_edge)
        # .filter(pl.col('pred').is_in(pheno_rel))
        .with_columns(
            id=pl.col('sub').str.split('/').list.last(),
            phenotype=pl.col('obj').str.split('/').list.last(),
            xrefs=pl.col('meta').struct.field('basicPropertyValues').list.eval(pl.element().struct.field('val')),
        )
    )

    # get phenotypes
    phenotype_xrefs_by_id = (
        mondo_phenotypes.filter(pl.col('xrefs').is_not_null())
        .select('id', pl.col('xrefs').alias('phenotype_xrefs'))
        .explode('phenotype_xrefs')
        .group_by('id')
        .agg(phenotype_xrefs=pl.col('phenotype_xrefs').unique())
    )

    # extract the short id from mondo_nodes for joining
    mondo_nodes_with_short_id = mondo_nodes.with_columns(short_id=pl.col('id').str.split('/').list.last())

    # then modify mondo_clean definition to join on the short id
    mondo_clean = (
        mondo_nodes_with_short_id.filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        .unnest('meta')
        .join(
            phenotype_xrefs_by_id,
            left_on='short_id',
            right_on='id',
            how='left',
        )
        .with_columns(
            diseaseFromSourceId=pl.col('short_id'),
            xrefs=pl.concat_list([
                pl.col('xrefs')
                .list.eval(pl.element().struct.field('val').str.replace(':', '_').unique())
                .fill_null([]),
                pl.col('definition').struct.field('xrefs').list.eval(pl.element().str.replace(':', '_')).fill_null([]),
                pl.when(pl.col('phenotype_xrefs').is_not_null())
                .then(pl.col('phenotype_xrefs').list.eval(pl.element().str.replace(':', '_')))
                .otherwise(pl.lit([])),
            ]),
            resource=pl.lit('MONDO'),
            diseaseFromSource=pl.col('lbl'),
            id=pl.concat_list([
                pl.col('xrefs').list.eval(pl.element().struct.field('val').unique()).fill_null([]),
                pl.col('definition').struct.field('xrefs').fill_null([]),
                pl.col('phenotype_xrefs').fill_null([]),
            ]),
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
            'short_id',
            'phenotype_xrefs',
        )
    )

    # explode by dbxrefs, filtering only omim and orphanet
    explode_mondo = (
        mondo_clean.explode('id').with_columns(
            pl.col('id').str.replace('Orphanet:', 'ORPHA:'),
        )
    ).explode('xrefs')

    # filter the mondo ids, leave only the ones that have an xref existing in efo
    trim_mondo = explode_mondo.join(
        df_disease.select('id', 'name'),
        left_on='xrefs',
        right_on='id',
        how='right',
    ).rename({'id_right': 'disease', 'name': 'diseaseName'})

    # then, filter again by the mondo ids that have phenotypes
    trim_mondo = (
        trim_mondo.join(
            mondo_phenotypes,
            left_on='diseaseFromSourceId',
            right_on='id',
            how='inner',
        )
        .drop('id', 'sub', 'pred', 'obj', 'meta')
        .with_columns(
            qualifierNot=pl.lit(False),
        )
    )

    ############################################################################
    # phenotypes dataset

    # prepare the phenotypes dataframe
    # straightfoward, just replace some colons and column names
    # also, compute qualifierNot column, which is just ~qualifier
    phenotypes = df_phenotype.with_columns(
        id=pl.col('database_id'),
        phenotype=pl.col('hpo_id').str.replace(':', '_'),
        aspect=pl.col('aspect'),
        bioCuration=pl.col('biocuration'),
        diseaseFromSourceId=pl.col('database_id'),
        diseaseFromSource=pl.col('disease_name'),
        evidenceType=pl.col('evidence'),
        frequency=pl.col('frequency'),
        modifiers=pl.col('modifier').str.split(';'),
        onset=pl.col('onset').str.split(';'),
        qualifier=pl.col('qualifier'),
        qualifierNot=pl.when(pl.col('qualifier').is_not_null())
        .then(
            pl.col('qualifier') == 'NOT',
        )
        .otherwise(
            pl.lit(False),
        ),
        references=pl.col('reference').str.split(';'),
        sex=pl.col('sex'),
        resource=pl.lit('HPO'),
    ).drop(
        'database_id',
        'disease_name',
        'hpo_id',
        'reference',
        'evidence',
        'modifier',
        'biocuration',
    )

    # join disease id into the phenotypes dataframe
    # the idea is to figure out which diseases have any xref or obsolete xref
    # that matches the database_id in the phenotype.hpoa, which are always
    # either omim or orphanet
    cut_disease = df_disease.select(
        pl.col('id'),
        pl.col('dbXRefs'),
        pl.col('obsoleteXRefs'),
        pl.col('name').alias('diseaseName'),
    )

    explode_disease = (
        cut_disease.with_columns(
            XRefs=pl.concat_list(
                pl.col('dbXRefs').fill_null([]),
                pl.col('obsoleteXRefs').fill_null([]),
            ),
        )
        .drop('dbXRefs', 'obsoleteXRefs')
        .explode('XRefs')
    )

    phenotypes_with_disease = phenotypes.join(
        explode_disease,
        left_on='id',
        right_on='XRefs',
        how='inner',
    ).rename({'id_right': 'disease'})

    # merge the phenotype.hpoa phenotypes with the mondo phenotypes
    # merged_phenotypes = pl.concat([phenotypes_with_disease, trim_mondo], how='diagonal')

    # grouping
    # join all the rows with the same disease id, by creating a list of
    # evidences that contain structs with most fields
    structed_phenotypes = phenotypes_with_disease.select(
        pl.col('disease'),
        pl.col('phenotype'),
        pl.struct(
            pl.col('aspect'),
            pl.col('bioCuration'),
            pl.col('diseaseFromSourceId'),
            pl.col('diseaseFromSource'),
            pl.col('diseaseName'),
            pl.col('evidenceType'),
            pl.col('frequency').str.replace(':', '_'),
            pl.col('modifiers').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
            pl.col('onset').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
            pl.col('qualifier'),
            pl.col('qualifierNot'),
            pl.col('references').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
            pl.col('sex'),
            pl.col('resource'),
        ).alias('evidence'),
    )

    grouped_phenotypes = structed_phenotypes.group_by(
        'disease',
        'phenotype',
    ).agg(
        pl.col('evidence'),
    )

    # write the result locally
    # raise NotImplementedError
    grouped_phenotypes.write_parquet(destination)
    logger.info('transformation complete')
