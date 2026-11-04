"""Schemas used by the pharmacovigilance transformer."""

import polars as pl

faers_schema = pl.Schema({
    'results': pl.List(
        pl.Struct({
            'safetyreportid': pl.String(),
            'seriousnessdeath': pl.String(),
            'primarysource': pl.Struct({
                'qualification': pl.String(),
            }),
            'patient': pl.Struct({
                'reaction': pl.List(
                    pl.Struct({
                        'reactionmeddrapt': pl.String(),
                    })
                ),
                'drug': pl.List(
                    pl.Struct({
                        'drugcharacterization': pl.String(),
                        'medicinalproduct': pl.String(),
                        'openfda': pl.Struct({
                            'brand_name': pl.List(pl.String()),
                            'generic_name': pl.List(pl.String()),
                            'substance_name': pl.List(pl.String()),
                        }),
                    })
                ),
            }),
        })
    )
})

pharmacovigilance_schema = pl.Schema({
    'chembl_id': pl.String(),
    'event': pl.String(),
    'count': pl.Int64(),
    'llr': pl.Float64(),
    'critval': pl.Float64(),
    'meddraCode': pl.String(),
})
