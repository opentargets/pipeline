"""Schemas for the CredibleSet and CredibleSetVariant datasets, and the vocabularies they type columns with.

`FinemappingMethod` and `Chromosome` are the fixed vocabularies of `fineMappingMethod` and
`chromosome`: the validated and published sets type those columns as enums of them, and set QC
flags a set outside them.
"""

from __future__ import annotations

from enum import StrEnum

import polars as pl

from pts.schemas.study import StudyType


class FinemappingMethod(StrEnum):
    """Fine-mapping methods."""

    PICS = 'PICS'
    SUSIE = 'SuSiE'
    SUSIE_INF = 'SuSiE-inf'

    @classmethod
    def susie(cls) -> tuple[FinemappingMethod, ...]:
        """The methods of the SuSiE family."""
        return (cls.SUSIE, cls.SUSIE_INF)


class Chromosome(StrEnum):
    """Chromosomes a credible set or a variant may lie on, in genome order.

    These are the Ensembl canonical chromosomes, which the genes and the gnomAD variants that
    credible sets are joined to lie on. There is no XY: a pseudoautosomal variant is on X or Y.
    """

    CHR_1 = '1'
    CHR_2 = '2'
    CHR_3 = '3'
    CHR_4 = '4'
    CHR_5 = '5'
    CHR_6 = '6'
    CHR_7 = '7'
    CHR_8 = '8'
    CHR_9 = '9'
    CHR_10 = '10'
    CHR_11 = '11'
    CHR_12 = '12'
    CHR_13 = '13'
    CHR_14 = '14'
    CHR_15 = '15'
    CHR_16 = '16'
    CHR_17 = '17'
    CHR_18 = '18'
    CHR_19 = '19'
    CHR_20 = '20'
    CHR_21 = '21'
    CHR_22 = '22'
    CHR_X = 'X'
    CHR_Y = 'Y'
    CHR_MT = 'MT'


credible_set_schema = pl.Schema({
    'id': pl.String(),
    'chromosome': pl.String(),
    'locusStart': pl.Int32(),
    'locusEnd': pl.Int32(),
    'studyId': pl.String(),
    'studyType': pl.String(),
    'leadVariantId': pl.String(),
    'fineMappingMethod': pl.String(),
    'logBayesFactor': pl.Float64(),
    'confidence': pl.Int8(),
    'purityMeanR2': pl.Float64(),
    'purityMinR2': pl.Float64(),
    'qualityControls': pl.List(pl.String()),
})
"""The CredibleSet schema: one row per credible set, keyed by `id`."""

validated_credible_set_schema = pl.Schema({
    **credible_set_schema,
    'chromosome': pl.Enum(Chromosome),
    'studyType': pl.Enum(StudyType),
    'fineMappingMethod': pl.Enum(FinemappingMethod),
})
"""The CredibleSet schema of the validated and published sets, with the columns of a fixed vocabulary as enums.

Spark reads the enum columns as strings, so the published dataset keeps the CredibleSet schema.
"""

credible_set_variant_schema = pl.Schema({
    'credibleSetId': pl.String(),
    'variantId': pl.String(),
    'pValueMantissa': pl.Float32(),
    'pValueExponent': pl.Int32(),
    'effectAlleleFrequency': pl.Float32(),
    'sampleSize': pl.Int32(),
    'reportedBeta': pl.Float64(),
    'reportedStandardError': pl.Float64(),
    'posteriorProbability': pl.Float32(),
    'logBayesFactor': pl.Float32(),
    'r2Overall': pl.Float32(),
    'conditionalBeta': pl.Float32(),
    'conditionalStandardError': pl.Float32(),
    'rescaledAbsBeta': pl.Float32(),
    'rescaledStandardError': pl.Float32(),
    'rescaledDirection': pl.Int8(),
    'qualityControls': pl.List(pl.String()),
})
"""The CredibleSetVariant schema: one row per variant of a credible set, keyed by `credibleSetId` and `variantId`."""

cross_set_flags_schema = pl.Schema({
    'id': pl.String(),
    'qualityControls': pl.List(pl.String()),
})
"""One row per credible set flagged by the cross-set QC: its identifier and its cross-set flags."""

credible_set_required = ('id', 'studyId', 'chromosome', 'locusStart', 'locusEnd', 'leadVariantId', 'fineMappingMethod')
"""The CredibleSet columns that must not be null: the id and the columns it is hashed from."""

credible_set_variant_required = ('credibleSetId', 'variantId', 'posteriorProbability')
"""The CredibleSetVariant columns that must not be null: the keys of a row and its posterior probability."""
