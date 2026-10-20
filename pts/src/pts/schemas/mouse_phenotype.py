"""Schemas of the mouse phenotype dataset."""

import polars as pl

BIOLOGICAL_MODEL = pl.Struct({
    'allelicComposition': pl.String,
    'geneticBackground': pl.String,
    'id': pl.String,
    'literature': pl.List(pl.String),
})

PHENOTYPE_CLASS = pl.Struct({'id': pl.String, 'label': pl.String})

#: The published mouse phenotype dataset; the excluded rows have `targetFromSourceId` first.
MOUSE_PHENOTYPE = pl.Schema({
    'targetInModel': pl.String,
    'targetInModelMgiId': pl.String,
    'targetInModelEnsemblId': pl.String,
    'targetFromSourceId': pl.String,
    'modelPhenotypeId': pl.String,
    'modelPhenotypeLabel': pl.String,
    'biologicalModels': pl.List(BIOLOGICAL_MODEL),
    'modelPhenotypeClasses': pl.List(PHENOTYPE_CLASS),
})
