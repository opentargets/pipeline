"""Schemas used in the gene burden evidence."""

import polars as pl

#: The evidence of every sub-source after disease mapping, as the old intermediate.
GENE_BURDEN_UNION = pl.Schema({
    'datasourceId': pl.String,
    'datatypeId': pl.String,
    'allelicRequirements': pl.List(pl.String),
    'targetFromSourceId': pl.String,
    'diseaseFromSource': pl.String,
    'pValueMantissa': pl.Float64,
    'pValueExponent': pl.Int32,
    'beta': pl.Float64,
    'betaConfidenceIntervalLower': pl.Float64,
    'betaConfidenceIntervalUpper': pl.Float64,
    'oddsRatio': pl.Float64,
    'oddsRatioConfidenceIntervalLower': pl.Float64,
    'oddsRatioConfidenceIntervalUpper': pl.Float64,
    'resourceScore': pl.Float64,
    'ancestry': pl.String,
    'ancestryId': pl.String,
    'literature': pl.List(pl.String),
    'projectId': pl.String,
    'cohortId': pl.String,
    'releaseVersion': pl.String,
    'studySampleSize': pl.Int32,
    'studyCases': pl.Int32,
    'studyCasesWithQualifyingVariants': pl.Int32,
    'statisticalMethod': pl.String,
    'statisticalMethodOverview': pl.String,
    'urls': pl.List(pl.Struct({'url': pl.String, 'niceName': pl.String})),
    'sex': pl.List(pl.String),
    'diseaseFromSourceId': pl.String,
    'diseaseFromSourceMappedId': pl.String,
})

#: What every sub-source parser returns: `GENE_BURDEN_UNION` before disease mapping, with the id
#: the source itself gives, if any, in place of the mapped one.
GENE_BURDEN_PARSED = pl.Schema({
    'curatedDiseaseFromSourceMappedId' if name == 'diseaseFromSourceMappedId' else name: dtype
    for name, dtype in GENE_BURDEN_UNION.items()
})

#: The curated evidence file's columns by position; their header names are not read.
CURATED_INPUT_COLUMNS = [
    'projectId',
    'targetFromSource',
    'targetFromSourceId',
    'diseaseFromSource',
    'diseaseFromSourceMappedId',
    'resourceScore',
    'pValueMantissa',
    'pValueExponent',
    'oddsRatio',
    'ConfidenceIntervalLower',
    'ConfidenceIntervalUpper',
    'beta',
    'sex',
    'ancestry',
    'ancestryId',
    'cohortId',
    'studySampleSize',
    'studyCases',
    'studyCasesWithQualifyingVariants',
    'allelicRequirements',
    'studyId',
    'statisticalMethod',
    'statisticalMethodOverview',
    'literature',
    'url',
]
