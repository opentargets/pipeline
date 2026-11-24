"""The pts schemas of three views before they had models (pts at d1a58968), copied: datamodels cannot
import pts.

`association_timeseries_schema()` (`pts/schemas/association.py`),
`OUTPUT_SCHEMA` (`pts/tests/test_coding_variant.py`) and `TARGET_VIEW` (`pts/schemas/target_view.py`).
"""

import polars as pl

ID_SOURCE = pl.Struct({'id': pl.String, 'source': pl.String})
LABEL_SOURCE = pl.Struct({'label': pl.String, 'source': pl.String})
LOCATION = pl.Struct({
    'location': pl.String,
    'source': pl.String,
    'termSL': pl.String,
    'labelSL': pl.String,
    'targetModifier': pl.String,
})
GENOMIC_LOCATION = pl.Struct({'chromosome': pl.String, 'start': pl.Int64, 'end': pl.Int64, 'strand': pl.Int32})
CANONICAL_TRANSCRIPT = pl.Struct({
    'id': pl.String,
    'chromosome': pl.String,
    'start': pl.Int64,
    'end': pl.Int64,
    'strand': pl.Int32,
})
GO = pl.Struct({
    'id': pl.String,
    'source': pl.String,
    'evidence': pl.String,
    'aspect': pl.String,
    'geneProduct': pl.String,
    'ecoId': pl.String,
})
HALLMARK_ATTRIBUTE = pl.Struct({'pmid': pl.Int64, 'description': pl.String, 'name': pl.String})
CANCER_HALLMARK = pl.Struct({'pmid': pl.Int64, 'description': pl.String, 'impact': pl.String, 'label': pl.String})
HALLMARKS = pl.Struct({'attributes': pl.List(HALLMARK_ATTRIBUTE), 'cancerHallmarks': pl.List(CANCER_HALLMARK)})
TARGET_CLASS = pl.Struct({'id': pl.Int64, 'label': pl.String, 'level': pl.String})
CONSTRAINT = pl.Struct({
    'constraintType': pl.String,
    'score': pl.Float32,
    'exp': pl.Float32,
    'obs': pl.Int32,
    'oe': pl.Float32,
    'oeLower': pl.Float32,
    'oeUpper': pl.Float32,
    'upperRank': pl.Int32,
    'upperBin': pl.Int32,
    'upperBin6': pl.Int32,
})
PATHWAY = pl.Struct({'pathwayId': pl.String, 'pathway': pl.String, 'topLevelTerm': pl.String})

TARGET = pl.Schema({
    'id': pl.String,
    'approvedSymbol': pl.String,
    'biotype': pl.String,
    'genomicLocation': GENOMIC_LOCATION,
    'approvedName': pl.String,
    'go': pl.List(GO),
    'hallmarks': HALLMARKS,
    'synonyms': pl.List(LABEL_SOURCE),
    'symbolSynonyms': pl.List(LABEL_SOURCE),
    'nameSynonyms': pl.List(LABEL_SOURCE),
    'functionDescriptions': pl.List(pl.String),
    'subcellularLocations': pl.List(LOCATION),
    'targetClass': pl.List(TARGET_CLASS),
    'obsoleteSymbols': pl.List(LABEL_SOURCE),
    'obsoleteNames': pl.List(LABEL_SOURCE),
    'constraint': pl.List(CONSTRAINT),
    'proteinIds': pl.List(ID_SOURCE),
    'dbXrefs': pl.List(ID_SOURCE),
    'pathways': pl.List(PATHWAY),
    'tss': pl.Int64,
})

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

TARGET_VIEW = pl.Schema({**TARGET, **VIEW_COLUMNS})

ASSOCIATION_TIMESERIES = pl.Schema({
    'diseaseId': pl.String(),
    'targetId': pl.String(),
    'aggregationType': pl.String(),
    'aggregationValue': pl.String(),
    'year': pl.Int32(),
    'associationScore': pl.Float64(),
    'novelty': pl.Float64(),
    'yearlyEvidenceCount': pl.Int64(),
    'isDirect': pl.Boolean(),
    'partitionKey': pl.String(),
})

DATASOURCE = pl.Struct({'datasourceCount': pl.Int32(), 'datasourceId': pl.String(), 'datasourceNiceName': pl.String()})

CODING_VARIANT = pl.Schema({
    'targetId': pl.String,
    'variantId': pl.String,
    'therapeuticAreas': pl.List(pl.String),
    'diseases': pl.List(pl.String),
    'datasources': pl.List(DATASOURCE),
    'variantEffect': pl.Float32,
    'variantFunctionalConsequenceIds': pl.List(pl.String),
    'uniprotAccessions': pl.List(pl.String),
    'referenceAminoAcid': pl.String,
    'aminoAcidPosition': pl.Int32,
    'alternateAminoAcid': pl.String,
})
