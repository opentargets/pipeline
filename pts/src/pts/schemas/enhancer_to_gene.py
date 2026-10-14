"""Schema for the EnhancerToGene dataset."""

import polars as pl

enhancer_to_gene_schema = pl.Schema({
    'chromosome': pl.String(),
    'start': pl.Int64(),
    'end': pl.Int64(),
    'geneId': pl.String(),
    'score': pl.Float64(),
    'distanceToTss': pl.Int32(),
    'resourceScore': pl.List(
        pl.Struct({
            'name': pl.String(),
            'value': pl.Float32(),
        })
    ),
    'datasourceId': pl.String(),
    'intervalType': pl.String(),
    'pmid': pl.String(),
    'biosampleName': pl.String(),
    'biosampleFromSourceId': pl.String(),
    'biosampleId': pl.String(),
    'studyId': pl.String(),
    'intervalId': pl.String(),
    'qualityControls': pl.List(pl.String()),
})
