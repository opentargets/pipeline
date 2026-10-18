"""Schemas used in the pharmacogenetics step."""

import polars as pl

#: The published `output/pharmacogenomics` schema.
PHARMACOGENOMICS = pl.Schema({
    'datasourceId': pl.String,
    'datasourceVersion': pl.String,
    'datatypeId': pl.String,
    'directionality': pl.String,
    'evidenceLevel': pl.String,
    'genotype': pl.String,
    'genotypeAnnotationText': pl.String,
    'genotypeId': pl.String,
    'haplotypeFromSourceId': pl.String,
    'haplotypeId': pl.String,
    'literature': pl.List(pl.String),
    'pgxCategory': pl.String,
    'phenotypeFromSourceId': pl.String,
    'phenotypeText': pl.String,
    'variantAnnotation': pl.List(
        pl.Struct({
            'baseAlleleOrGenotype': pl.String,
            'comparisonAlleleOrGenotype': pl.String,
            'directionality': pl.String,
            'effect': pl.String,
            'effectDescription': pl.String,
            'effectType': pl.String,
            'entity': pl.String,
            'id': pl.String,
            'literature': pl.String,
        })
    ),
    'studyId': pl.String,
    'targetFromSourceId': pl.String,
    'variantFunctionalConsequenceId': pl.String,
    'variantRsId': pl.String,
    'variantId': pl.String,
    'isDirectTarget': pl.Boolean,
    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String, 'drugId': pl.String})),
})

_DERIVED = {'variantId', 'isDirectTarget', 'drugs'}

#: The ClinPGx annotation columns the step reads; any absent from the file is read as null.
CLINPGX_SCHEMA = pl.Schema({
    **{name: dtype for name, dtype in PHARMACOGENOMICS.items() if name not in _DERIVED},
    'drugs': pl.List(pl.Struct({'drugFromSource': pl.String})),
})

#: The curated phenotypes lookup.
PHENOTYPES_SCHEMA = pl.Schema({'genotypeAnnotationText': pl.String, 'phenotypeText': pl.List(pl.String)})
