"""Credible set locus: the `credible_set_locus_view` view (pts `credible_set_view`).

One row per credible set with its variants, by `posteriorProbability` descending, then `variantId`.
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, StructModel, from_output
from ot_datamodels.outputs.credible_set_variant import CredibleSetVariant
from ot_datamodels.serving import Serving

CredibleSetId = from_output(CredibleSetVariant, 'credibleSetId', primary_key=True)
VariantId = from_output(CredibleSetVariant, 'variantId')
PosteriorProbability = from_output(CredibleSetVariant, 'posteriorProbability')
LogBayesFactor = from_output(CredibleSetVariant, 'logBayesFactor')
PValueMantissa = from_output(CredibleSetVariant, 'pValueMantissa')
PValueExponent = from_output(CredibleSetVariant, 'pValueExponent')
EffectAlleleFrequency = from_output(CredibleSetVariant, 'effectAlleleFrequency')
SampleSize = from_output(CredibleSetVariant, 'sampleSize')
ReportedBeta = from_output(CredibleSetVariant, 'reportedBeta')
ReportedStandardError = from_output(CredibleSetVariant, 'reportedStandardError')
R2Overall = from_output(CredibleSetVariant, 'r2Overall')
ConditionalBeta = from_output(CredibleSetVariant, 'conditionalBeta')
ConditionalStandardError = from_output(CredibleSetVariant, 'conditionalStandardError')
RescaledAbsBeta = from_output(CredibleSetVariant, 'rescaledAbsBeta')
RescaledStandardError = from_output(CredibleSetVariant, 'rescaledStandardError')
RescaledDirection = from_output(CredibleSetVariant, 'rescaledDirection')

_EXAMPLE = CredibleSetVariant.example
#: The served row, also the order of `LocusRow`.
LOCUS_FIELDS = [
    'variantId',
    'posteriorProbability',
    'logBayesFactor',
    'pValueMantissa',
    'pValueExponent',
    'effectAlleleFrequency',
    'sampleSize',
    'reportedBeta',
    'reportedStandardError',
    'r2Overall',
    'conditionalBeta',
    'conditionalStandardError',
    'rescaledAbsBeta',
    'rescaledStandardError',
    'rescaledDirection',
    'qualityControls',
]


class LocusRow(StructModel):
    """`locus[]`: a credible set variant, `qualityControls` without LowCardinality."""

    variantId: VariantId
    posteriorProbability: PosteriorProbability
    logBayesFactor: LogBayesFactor
    pValueMantissa: PValueMantissa
    pValueExponent: PValueExponent
    effectAlleleFrequency: EffectAlleleFrequency
    sampleSize: SampleSize
    reportedBeta: ReportedBeta
    reportedStandardError: ReportedStandardError
    r2Overall: R2Overall
    conditionalBeta: ConditionalBeta
    conditionalStandardError: ConditionalStandardError
    rescaledAbsBeta: RescaledAbsBeta
    rescaledStandardError: RescaledStandardError
    rescaledDirection: RescaledDirection
    qualityControls: list[Annotated[str, Field(min_length=1)]] = Field(
        description=CredibleSetVariant.model_fields['qualityControls'].description
    )


class CredibleSetLocusView(DatasetModel):
    """Credible set locus."""

    dataset_name: ClassVar[str] = 'credible_set_locus_view'
    dataset_title: ClassVar[str] = 'Credible set locus'
    dataset_description: ClassVar[str] = 'The variants of each credible set, most probable first'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = {
        'credibleSetId': _EXAMPLE['credibleSetId'],
        'locus': [{k: _EXAMPLE[k] for k in LOCUS_FIELDS}],
    }

    credibleSetId: CredibleSetId
    locus: list[LocusRow] = Field(
        min_length=1, description='Variants of the credible set, by posteriorProbability descending, then variantId'
    )
