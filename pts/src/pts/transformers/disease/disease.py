"""Build the disease index from the EFO otar slim ontology."""

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.ontology import node
from pts.transformers.disease.coalescing import (
    annotate_name_duplicates,
    annotate_xref_duplicates,
    remap_edges,
    resolve_replacement_chains,
)
from pts.transformers.utils.dataset import write_dataset


def disease(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    # load the ontology
    logger.debug('loading efo')
    h = StorageHandle(source)
    f = h.open()
    initial = pl.read_json(f)  # ty:ignore[invalid-argument-type]

    logger.debug('starting transformation')

    # prepare dataframes
    n = pl.DataFrame(
        initial['graphs'][0][0]['nodes'],
        schema=node,
    )
    e = pl.DataFrame(
        initial['graphs'][0][0]['edges'],
    )

    # Mark duplicate nodes deprecated before any filtering: first the terms
    # sharing a label, then the cross-ontology equivalents whose labels differ.
    # Each pass is followed by resolving its replacement pointers and rewriting
    # edges onto retained identifiers.  Edges must be rewritten between the two
    # passes as well as after them, because the second pass reads the is_a
    # hierarchy: an unresolved merge hides ancestor links, and the hierarchy
    # guard would then let a parent and child merge.
    n = resolve_replacement_chains(annotate_name_duplicates(n))
    e = remap_edges(e, n)
    n = resolve_replacement_chains(annotate_xref_duplicates(n, e))
    e = remap_edges(e, n)

    # clean the nodes
    n_clean = (
        n
        .filter(
            pl.col('type') == 'CLASS',
            ~pl.col('meta').struct['deprecated'] | pl.col('meta').struct['deprecated'].is_null(),
        )
        .unnest('meta')
        .with_columns(
            id=pl.col('id').str.split('/').list.last(),
            code=pl.col('id'),
            name=pl.col('lbl'),
            isTherapeuticArea=pl.col('subsets').list.contains('"therapeutic_area"'),
            description=pl.col('definition').struct['val'],
            dbXRefs=pl
            .col('xrefs')
            .list.eval(pl.element().struct.field('val').unique())
            .fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
        )
        .drop(
            'basicPropertyValues',
            'comments',
            'definition',
            'deprecated',
            'lbl',
            'subsets',
            'type',
            'xrefs',
        )
    )

    # get parents, by filtering edges with 'is_a' predicate
    parents = (
        e
        .filter(pl.col('pred') == 'is_a')
        .with_columns(
            id=pl.col('sub').str.split('/').list.last(),
            parents=pl.col('obj').str.split('/').list.last(),
        )
        .group_by('id')
        .agg(pl.col('parents').drop_nulls())
    )
    n_parents = n_clean.join(parents, on='id', how='left').with_columns(
        parents=pl.col('parents').fill_null(pl.Series([[]], dtype=pl.List(pl.String)))
    )

    # get location_ids by filtering edges with 'located_in' predicate
    location_ids = (
        e
        .filter(pl.col('pred') == 'http://purl.obolibrary.org/obo/BFO_0000050')
        .with_columns(
            id=pl.col('sub').str.split('/').list.last(),
            directLocationIds=pl.col('obj').str.split('/').list.last(),
        )
        .group_by('id')
        .agg(pl.col('directLocationIds').drop_nulls())
        .drop('directLocationIds')
    )
    n_location_ids = n_parents.join(location_ids, on='id', how='left')

    # get synonyms by:
    # 1. exploding the synonyms column
    # 2. filtering the synonyms with the correct predicates
    # 3. cleaning the values (removing newlines and whitespaces)
    # 4. grouping by id and predicate
    # 5. pivoting the values into columns
    # 6. renaming columns to the simplified synonym schema
    # 7. selecting the columns
    synonym_predicates = [
        'hasExactSynonym',
        'hasRelatedSynonym',
        'hasNarrowSynonym',
        'hasBroadSynonym',
    ]
    synonym_rename_mapping = {
        'hasExactSynonym': 'exactSynonyms',
        'hasRelatedSynonym': 'relatedSynonyms',
        'hasNarrowSynonym': 'narrowSynonyms',
        'hasBroadSynonym': 'broadSynonyms',
    }
    synonym_columns = list(synonym_rename_mapping.values())
    synonyms = (
        n_location_ids['id', 'synonyms']
        .explode('synonyms')
        .filter(
            pl.col('synonyms').struct['pred'].is_in(synonym_predicates),
        )
        .unnest('synonyms')
        .with_columns(
            val=pl.col('val').str.replace_all('\n', '').str.strip_chars(),
        )
        .group_by([
            'id',
            'pred',
        ])
        .agg(pl.col('val').drop_nulls().unique())
        .pivot(
            values='val',
            index='id',
            columns='pred',  # ty:ignore[unknown-argument]
            aggregate_function='first',
        )  # ty:ignore[missing-argument]
        .with_columns(
            **{k: pl.col(k).fill_null([]) for k in synonym_predicates},
        )
        .rename(synonym_rename_mapping)
        .select(['id', *synonym_columns])
    )

    n_synonyms = (
        n_location_ids
        .drop('synonyms')
        .join(synonyms, on='id', how='left')
        .with_columns(
            **{col: pl.col(col).fill_null(pl.Series([[]], dtype=pl.List(pl.String))) for col in synonym_columns},
        )
    )

    # get obsolete ids by getting deprecated nodes with a 'IAO_0100001' predicate
    obsolete_ids = (
        n
        .unnest('meta')
        .explode('basicPropertyValues')
        .unnest('basicPropertyValues')
        .filter(
            pl.col('deprecated'),
            pl.col('pred') == 'http://purl.obolibrary.org/obo/IAO_0100001',
        )
        .select(
            pl.col('id'),
            pl.col('val').alias('code'),
        )
    )

    # Get the obsolete terms
    obsolete_terms = (
        obsolete_ids
        .with_columns(
            code=pl.col('code'),
            obsoleteTerms=pl.col('id').str.split('/').list.last(),
        )
        .group_by('code')
        .agg(pl.col('obsoleteTerms'))
    )

    # Get the xrefs for all the obsolete terms
    obsolete_xrefs = (
        n
        .unnest('meta')
        .filter(pl.col('xrefs').is_not_null())
        .select(pl.col('id'), pl.col('xrefs'))
        .join(obsolete_ids, on='id')
        .explode('xrefs')
        .unnest('xrefs')
        .select(pl.col('code'), pl.col('val'))
        .group_by('code')
        .agg(pl.col('val').alias('obsoleteXRefs'))
    )

    # join obsolete term list and obsolete xref list to the ids of the entities that
    # make them obsolete
    n_obsolete_terms = (
        n_synonyms
        .join(
            obsolete_terms,
            on='code',
            how='left',
        )
        .join(
            obsolete_xrefs,
            on='code',
            how='left',
        )
        .with_columns(
            obsoleteTerms=pl.col('obsoleteTerms').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
            obsoleteXRefs=pl.col('obsoleteXRefs').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
        )
    )

    # get children by exploding the parents column, making it the new id and
    # then aggregating by the old id
    children = (
        n_obsolete_terms
        .explode('parents')
        .filter(pl.col('parents').is_not_null())
        .group_by('parents')
        .agg(pl.col('id').alias('children'))
        .rename({'parents': 'id'})
    )
    n_children = n_obsolete_terms.join(children, on='id', how='left').with_columns(
        children=pl.col('children').fill_null(pl.Series([[]], dtype=pl.List(pl.String)))
    )

    # get ancestors and therapeutic areas:
    # 1. explode the parents column into direct relationships
    # 2. create two dataframes, one to accumulate ancestors and another for current level ancestors
    # 3. join the current level ancestors with their parents to get the next level ancestors,
    #    until there are no more entries in the next level
    # 4. group all ancestors by id
    # 5. get therapeutic area ancestors by filtering the ancestors that are therapeutic areas
    # 6. create self references for therapeutic areas (they are their own ancestors)
    # 7. group 5 and 6
    # 8. join both ancestors and therapeutic areas with the original dataframe
    direct_relationships = (
        n_children
        .select(['id', 'parents'])
        .filter(pl.col('parents').is_not_null())
        .explode('parents')
        .rename({'parents': 'ancestor'})
    )
    all_ancestors = direct_relationships.select(['id', 'ancestor'])
    current_level = direct_relationships.select(['id', 'ancestor'])

    while True:
        if current_level.height == 0:
            break

        next_level = (
            current_level
            .join(
                n_children.select(['id', 'parents']),
                left_on='ancestor',
                right_on='id',
            )
            .filter(pl.col('parents').is_not_null())
            .explode('parents')
            .select(pl.col('id'), pl.col('parents').alias('ancestor'))
        )

        if next_level.height == 0:
            break

        all_ancestors = pl.concat([all_ancestors, next_level])
        current_level = next_level

    ancestors_grouped = all_ancestors.group_by('id').agg(
        pl.col('ancestor').drop_nulls().unique().alias('ancestors'),
    )

    therapeutic_area_ancestors = (
        all_ancestors
        .join(
            n_children.select(['id', 'isTherapeuticArea']),
            left_on='ancestor',
            right_on='id',
        )
        .filter(pl.col('isTherapeuticArea'))
        .join(n_children.select('id'), left_on='ancestor', right_on='id', how='inner')
        .select(pl.col('id'), pl.col('ancestor'))
    )

    therapeutic_area_selfreferences = (
        n_children
        .filter(pl.col('isTherapeuticArea'))
        .with_columns(ancestor=pl.col('id'))
        .select(
            pl.col('id'),
            pl.col('ancestor'),
        )
    )

    all_therapeutic_area_ancestors = (
        pl
        .concat([
            therapeutic_area_ancestors,
            therapeutic_area_selfreferences,
        ])
        .group_by('id')
        .agg(pl.col('ancestor').drop_nulls().unique().alias('therapeuticAreas'))
    )

    n_ancestors = (
        n_children
        .join(
            ancestors_grouped,
            on='id',
            how='left',
        )
        .join(
            all_therapeutic_area_ancestors,
            on='id',
            how='left',
        )
        .with_columns(
            ancestors=pl.col('ancestors').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
        )
    )

    # get descendants by exploding the ancestors column, making it the new id and then aggregating by the old id
    descendants_grouped = (
        all_ancestors
        .select(
            pl.col('ancestor').alias('id'),
            pl.col('id').alias('descendant'),
        )
        .group_by('id')
        .agg(pl.col('descendant').drop_nulls().unique().alias('descendants'))
    )
    n_descendants = n_ancestors.join(
        descendants_grouped,
        on='id',
        how='left',
    ).with_columns(
        descendants=pl.col('descendants').fill_null(pl.Series([[]], dtype=pl.List(pl.String))),
    )

    # create the ontology struct by putting there some stuff already present outside
    n_ontology = n_descendants.with_columns(
        ontology=pl.struct(
            isTherapeuticArea=pl.col('isTherapeuticArea').fill_null(False),
            leaf=pl.col('descendants').is_null(),
            sources=pl.struct(
                url=pl.col('code'),
                name=pl.col('id'),
            ),
        )
    ).drop('isTherapeuticArea')

    disease_index = n_ontology.with_columns(
        synonyms=pl.struct(
            hasExactSynonym=pl.col('exactSynonyms'),
            hasRelatedSynonym=pl.col('relatedSynonyms'),
            hasNarrowSynonym=pl.col('narrowSynonyms'),
            hasBroadSynonym=pl.col('broadSynonyms'),
        )
    )

    # write the result
    write_dataset(disease_index, destination)
    logger.info('transformation complete')
