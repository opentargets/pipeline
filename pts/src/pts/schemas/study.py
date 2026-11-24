"""Schemas and study types of the Study dataset and its curation input."""

from typing import get_args

import polars as pl
from ot_datamodels.vocabularies import StudyType

#: The gentropy study types, in the model's order.
STUDY_TYPES: tuple[StudyType, ...] = get_args(StudyType)


study_schema = pl.Schema({
    'studyId': pl.String(),
    'projectId': pl.String(),
    'studyType': pl.String(),
    'traitFromSource': pl.String(),
    'traitFromSourceMappedIds': pl.List(pl.String()),
    'diseaseIds': pl.List(pl.String()),
    'geneId': pl.String(),
    'biosampleFromSourceId': pl.String(),
    'biosampleId': pl.String(),
    'pubmedId': pl.String(),
    'publicationTitle': pl.String(),
    'publicationFirstAuthor': pl.String(),
    'publicationDate': pl.String(),
    'publicationJournal': pl.String(),
    'backgroundTraitFromSourceMappedIds': pl.List(pl.String()),
    'backgroundDiseaseIds': pl.List(pl.String()),
    'initialSampleSize': pl.String(),
    'nCases': pl.Int32(),
    'nControls': pl.Int32(),
    'nSamples': pl.Int32(),
    'cohorts': pl.List(pl.String()),
    'ldPopulationStructure': pl.List(
        pl.Struct({
            'ldPopulation': pl.String(),
            'relativeSampleSize': pl.Float64(),
        })
    ),
    'discoverySamples': pl.List(
        pl.Struct({
            'sampleSize': pl.Int32(),
            'ancestry': pl.String(),
        })
    ),
    'replicationSamples': pl.List(
        pl.Struct({
            'sampleSize': pl.Int32(),
            'ancestry': pl.String(),
        })
    ),
    'qualityControls': pl.List(pl.String()),
    'analysisFlags': pl.List(pl.String()),
    'summarystatsLocation': pl.String(),
    'hasSumstats': pl.Boolean(),
    'condition': pl.String(),
    'sumstatQCValues': pl.List(
        pl.Struct({
            'QCCheckName': pl.String(),
            'QCCheckValue': pl.Float32(),
        })
    ),
})

#: The curation file of superseded studies: each study and the study that supersedes it.
superseded_studies_schema = pl.Schema({'studyId': pl.String(), 'supersededBy': pl.String()})
