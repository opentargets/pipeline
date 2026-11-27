"""Credible set view: the `credible_set_view` view (pts `credible_set_view`).

One row per credible set with the statistics of its lead variant, the number of its variants and whether it is a
trans QTL. A set is served only with its lead variant's row. Also holds the column types the credible-set read
models (`study_credible_set_view`, `target_credible_set_view`, `variant_credible_set_view`) share.
"""

from typing import Annotated, ClassVar

import polars as pl
from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, PolarsDtype, from_output
from ot_datamodels.outputs.credible_set import CredibleSet
from ot_datamodels.outputs.credible_set_variant import CredibleSetVariant
from ot_datamodels.serving import Serving

CredibleSetId = from_output(CredibleSet, 'id', primary_key=True)
StudyId = from_output(CredibleSet, 'studyId')
StudyType = from_output(CredibleSet, 'studyType')
Chromosome = from_output(CredibleSet, 'chromosome')
LeadVariantId = from_output(CredibleSet, 'leadVariantId')
LeadPValueMantissa = from_output(CredibleSetVariant, 'pValueMantissa')
LeadPValueExponent = from_output(CredibleSetVariant, 'pValueExponent')
LeadBeta = from_output(CredibleSetVariant, 'reportedBeta')
LeadStandardError = from_output(CredibleSetVariant, 'reportedStandardError')
LeadEffectAlleleFrequency = from_output(CredibleSetVariant, 'effectAlleleFrequency')
LeadSampleSize = from_output(CredibleSetVariant, 'sampleSize')
LocusStart = from_output(CredibleSet, 'locusStart')
LocusEnd = from_output(CredibleSet, 'locusEnd')
FineMappingMethod = from_output(CredibleSet, 'fineMappingMethod')
LogBayesFactor = from_output(CredibleSet, 'logBayesFactor')
Confidence = from_output(CredibleSet, 'confidence')
PurityMeanR2 = from_output(CredibleSet, 'purityMeanR2')
PurityMinR2 = from_output(CredibleSet, 'purityMinR2')
QualityControls = from_output(CredibleSet, 'qualityControls')

# shared by the read models: values with a default or a type no output column has
LeadAllele = Annotated[
    str, Field(description="Allele of the lead variant; '' when the lead variant is not in the variant dataset")
]
ApprovedSymbol = Annotated[
    Annotated[str, ClickHouseType('LowCardinality(String)')] | None,
    Field(min_length=1, description='Approved symbol of the target; null when the gene is not a target'),
]

_EXAMPLE = CredibleSet.example
_LEAD = CredibleSetVariant.example


class CredibleSetView(DatasetModel):
    """Credible set view."""

    dataset_name: ClassVar[str] = 'credible_set_view'
    dataset_title: ClassVar[str] = 'Credible set view'
    dataset_description: ClassVar[str] = (
        'Credible sets with the statistics of their lead variant and the number of their variants'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    clickhouse: ClassVar[Serving] = Serving(engine='rocksdb')
    example: ClassVar[dict] = {
        'credibleSetId': _EXAMPLE['id'],
        'studyId': _EXAMPLE['studyId'],
        'studyType': _EXAMPLE['studyType'],
        'chromosome': _EXAMPLE['chromosome'],
        'leadVariantId': _EXAMPLE['leadVariantId'],
        'leadPosition': 44411929,
        'leadPValueMantissa': _LEAD['pValueMantissa'],
        'leadPValueExponent': _LEAD['pValueExponent'],
        'leadBeta': _LEAD['reportedBeta'],
        'leadStandardError': _LEAD['reportedStandardError'],
        'leadEffectAlleleFrequency': _LEAD['effectAlleleFrequency'],
        'leadSampleSize': _LEAD['sampleSize'],
        'locusStart': _EXAMPLE['locusStart'],
        'locusEnd': _EXAMPLE['locusEnd'],
        'locusSize': 1,
        'fineMappingMethod': _EXAMPLE['fineMappingMethod'],
        'logBayesFactor': _EXAMPLE['logBayesFactor'],
        'confidence': _EXAMPLE['confidence'],
        'purityMeanR2': _EXAMPLE['purityMeanR2'],
        'purityMinR2': _EXAMPLE['purityMinR2'],
        'qualityControls': _EXAMPLE['qualityControls'],
        'isTransQtl': 0,
    }

    credibleSetId: CredibleSetId
    studyId: StudyId
    studyType: StudyType
    chromosome: Chromosome
    leadVariantId: LeadVariantId
    leadPosition: Annotated[int, PolarsDtype(pl.UInt32)] = Field(
        ge=0, description='Position of the lead variant, from its identifier'
    )
    leadPValueMantissa: LeadPValueMantissa
    leadPValueExponent: LeadPValueExponent
    leadBeta: LeadBeta
    leadStandardError: LeadStandardError
    leadEffectAlleleFrequency: LeadEffectAlleleFrequency
    leadSampleSize: LeadSampleSize
    locusStart: LocusStart
    locusEnd: LocusEnd
    locusSize: Annotated[int, PolarsDtype(pl.UInt32)] = Field(
        ge=1, description='Number of variants in the credible set'
    )
    fineMappingMethod: FineMappingMethod
    logBayesFactor: LogBayesFactor
    confidence: Confidence
    purityMeanR2: PurityMeanR2
    purityMinR2: PurityMinR2
    qualityControls: QualityControls
    isTransQtl: Annotated[int, PolarsDtype(pl.UInt8)] = Field(
        ge=0, le=1, description='1 if the credible set is flagged as a trans QTL, else 0'
    )
