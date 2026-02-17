"""Schemas used in the disease dataset."""

import polars as pl

synonym_struct_schema = pl.Schema({
    'hasExactSynonym': pl.List(pl.String()),
    'hasRelatedSynonym': pl.List(pl.String()),
    'hasBroadSynonym': pl.List(pl.String()),
    'hasNarrowSynonym': pl.List(pl.String()),
})

schema = pl.Schema({
    'id': pl.String(),
    'code': pl.String(),
    'name': pl.String(),
    'description': pl.String(),
    'dbXRefs': pl.List(pl.String()),
    'parents': pl.List(pl.String()),
    'exactSynonyms': pl.List(pl.String()),
    'relatedSynonyms': pl.List(pl.String()),
    'narrowSynonyms': pl.List(pl.String()),
    'broadSynonyms': pl.List(pl.String()),
    'synonyms': pl.Struct(synonym_struct_schema),
    'obsoleteTerms': pl.List(pl.String()),
    'obsoleteXRefs': pl.List(pl.String()),
    'children': pl.List(pl.String()),
    'ancestors': pl.List(pl.String()),
    'therapeuticAreas': pl.List(pl.String()),
    'descendants': pl.List(pl.String()),
    'ontology': pl.Struct({
        'isTherapeuticArea': pl.Boolean(),
        'leaf': pl.Boolean(),
        'sources': pl.Struct({
            'url': pl.String(),
            'name': pl.String(),
        }),
    }),
})
