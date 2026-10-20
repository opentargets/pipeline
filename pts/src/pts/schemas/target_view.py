"""Schemas of the target view dataset."""

import polars as pl

from pts.schemas.target import CANONICAL_TRANSCRIPT, TARGET

HOMOLOGUE = pl.Struct({
    'speciesId': pl.String,
    'speciesName': pl.String,
    'homologyType': pl.String,
    'targetGeneId': pl.String,
    'isHighConfidence': pl.String,
    'targetGeneSymbol': pl.String,
    'queryPercentageIdentity': pl.Float64,
    'targetPercentageIdentity': pl.Float64,
    'priority': pl.Int32,
})

TRACTABILITY = pl.Struct({'modality': pl.String, 'id': pl.String, 'value': pl.Boolean})

SAFETY_LIABILITY = pl.Struct({
    'event': pl.String,
    'eventId': pl.String,
    'effects': pl.List(pl.Struct({'direction': pl.String, 'dosing': pl.String})),
    'biosamples': pl.List(
        pl.Struct({
            'tissueLabel': pl.String,
            'tissueId': pl.String,
            'cellLabel': pl.String,
            'cellFormat': pl.String,
            'cellId': pl.String,
        })
    ),
    'datasource': pl.String,
    'literature': pl.String,
    'url': pl.String,
    'studies': pl.List(pl.Struct({'description': pl.String, 'name': pl.String, 'type': pl.String})),
})

CHEMICAL_PROBE = pl.Struct({
    'targetFromSourceId': pl.String,
    'id': pl.String,
    'drugFromSourceId': pl.String,
    'drugId': pl.String,
    'mechanismOfAction': pl.List(pl.String),
    'origin': pl.List(pl.String),
    'control': pl.String,
    'isHighQuality': pl.Boolean,
    'probesDrugsScore': pl.Float64,
    'scoreInCells': pl.Float64,
    'scoreInOrganisms': pl.Float64,
    'urls': pl.List(pl.Struct({'niceName': pl.String, 'url': pl.String})),
})

TRANSCRIPT = pl.Struct({
    'transcriptId': pl.String,
    'biotype': pl.String,
    'uniprotId': pl.String,
    'isUniprotReviewed': pl.Boolean,
    'translationId': pl.String,
    'alphafoldId': pl.String,
    'uniprotIsoformId': pl.String,
    'isEnsemblCanonical': pl.Boolean,
})

#: The columns the view adds after the target columns.
VIEW_COLUMNS = pl.Schema({
    'homologues': pl.List(HOMOLOGUE),
    'tractability': pl.List(TRACTABILITY),
    'safetyLiabilities': pl.List(SAFETY_LIABILITY),
    'chemicalProbes': pl.List(CHEMICAL_PROBE),
    'transcriptIds': pl.List(pl.String),
    'transcripts': pl.List(TRANSCRIPT),
    'canonicalTranscript': CANONICAL_TRANSCRIPT,
    'canonicalExons': pl.List(pl.String),
    'alternativeGenes': pl.List(pl.String),
})

#: The published target view dataset.
TARGET_VIEW = pl.Schema({**TARGET, **VIEW_COLUMNS})
