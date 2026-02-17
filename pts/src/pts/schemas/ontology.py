"""Schemas used in the ontology dataset."""

import polars as pl

node_dict = {
    'id': pl.String(),
    'lbl': pl.String(),
    'meta': pl.Struct(
        {
            'basicPropertyValues': pl.List(
                pl.Struct(
                    {
                        'pred': pl.String(),
                        'val': pl.String(),
                    },
                ),
            ),
            'comments': pl.List(
                pl.String(),
            ),
            'definition': pl.Struct(
                {
                    'val': pl.String(),
                    'xrefs': pl.List(
                        pl.String(),
                    ),
                },
            ),
            'deprecated': pl.Boolean(),
            'subsets': pl.List(
                pl.String(),
            ),
            'synonyms': pl.List(
                pl.Struct(
                    {
                        'pred': pl.String(),
                        'synonymType': pl.String(),
                        'val': pl.String(),
                        'xrefs': pl.List(
                            pl.String(),
                        ),
                    },
                ),
            ),
            'xrefs': pl.List(
                pl.Struct(
                    {
                        'val': pl.String(),
                    },
                ),
            ),
        },
    ),
    'type': pl.String(),
}
node = pl.Schema(node_dict)

edge_dict = {
    'sub': pl.String(),
    'pred': pl.String(),
    'obj': pl.String(),
    'meta': pl.Struct({
        'xrefs': pl.List(
            pl.Struct({
                'val': pl.String(),
            }),
        ),
        'basicPropertyValues': pl.List(
            pl.Struct({
                'pred': pl.String(),
                'val': pl.String(),
            }),
        ),
    }),
}
edge = pl.Schema(edge_dict)


schema = pl.Schema({
    'graphs': pl.List(
        pl.Struct({
            'id': pl.String(),
            'meta': pl.Struct({
                'basicPropertyValues': pl.List(
                    pl.Struct({
                        'pred': pl.String(),
                        'val': pl.String(),
                    }),
                ),
                'version': pl.String(),
            }),
            'nodes': pl.List(pl.Struct(node_dict)),
            'edges': pl.List(pl.Struct(edge_dict)),
            'logicalDefinitionAxioms': pl.List(
                pl.Struct({
                    'definedClassId': pl.String(),
                    'genusIds': pl.List(pl.String()),
                    'restictions': pl.List(
                        pl.Struct({
                            'propertyId': pl.String,
                            'fillerId': pl.String,
                        }),
                    ),
                }),
            ),
            'domainRangeAxioms': pl.List(
                pl.Struct({
                    'predicateId': pl.String(),
                    'domainClassIds': pl.List(pl.String()),
                    'rangeClassIds': pl.List(pl.String()),
                }),
            ),
        }),
    ),
})
