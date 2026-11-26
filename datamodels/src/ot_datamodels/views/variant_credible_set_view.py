"""Variant credible set: the `variant_credible_set_view` view, served as `variant_credible_sets`.

Written by pts `credible_set_view`.

One row per credible set variant, for the variant page: the variant's posterior probability, its set's lead variant
and statistics, the set's study (trait, condition, target, biosample, diseases and their therapeutic areas) and its
top L2G gene (the highest score, then the lowest `geneId`; null without predictions).
"""

from typing import Annotated, ClassVar

from pydantic import Field

from ot_datamodels import ClickHouseType, DatasetModel, StructModel, from_output
from ot_datamodels.outputs.biosample import Biosample
from ot_datamodels.outputs.credible_set import CredibleSet
from ot_datamodels.outputs.credible_set_variant import CredibleSetVariant
from ot_datamodels.outputs.disease import Disease
from ot_datamodels.outputs.patterns import ONTOLOGY_ID
from ot_datamodels.outputs.study import Study
from ot_datamodels.views.credible_set_view import ApprovedSymbol, CredibleSetView, LeadAllele
from ot_datamodels.views.study_credible_set_view import StudyCredibleSetView

StudyType = from_output(CredibleSet, 'studyType')
VariantId = from_output(CredibleSetVariant, 'variantId')
CredibleSetId = from_output(CredibleSet, 'id')
PosteriorProbability = from_output(CredibleSetVariant, 'posteriorProbability')
LocusSize = from_output(CredibleSetView, 'locusSize')
PValueMantissa = from_output(CredibleSetVariant, 'pValueMantissa')
PValueExponent = from_output(CredibleSetVariant, 'pValueExponent')
Beta = from_output(CredibleSetVariant, 'reportedBeta')
FineMappingMethod = from_output(CredibleSet, 'fineMappingMethod')
Confidence = from_output(CredibleSet, 'confidence')
IsTransQtl = from_output(CredibleSetView, 'isTransQtl')
LeadVariantId = from_output(CredibleSet, 'leadVariantId')
LeadChromosome = from_output(CredibleSet, 'chromosome')
LeadPosition = from_output(CredibleSetView, 'leadPosition')
StudyId = from_output(CredibleSet, 'studyId')
StudyTargetId = from_output(Study, 'geneId')
BiosampleId = from_output(Study, 'biosampleFromSourceId')
TopL2GGeneId = from_output(StudyCredibleSetView, 'topL2GGeneId')
TopL2GScore = from_output(StudyCredibleSetView, 'topL2GScore')

_SET = CredibleSetView.example


class DiseaseName(StructModel):
    """A disease or therapeutic area: its id and name."""

    id: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        pattern=ONTOLOGY_ID, description=Disease.model_fields['id'].description
    )
    name: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        min_length=1, description=Disease.model_fields['name'].description
    )


class VariantCredibleSetView(DatasetModel):
    """Variant credible set."""

    dataset_name: ClassVar[str] = 'variant_credible_set_view'
    dataset_title: ClassVar[str] = 'Variant credible set'
    dataset_description: ClassVar[str] = (
        'The credible sets of each variant, with their lead variant, study and top L2G gene'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'studyType': _SET['studyType'],
        'variantId': _SET['leadVariantId'],
        'credibleSetId': _SET['credibleSetId'],
        'posteriorProbability': 1.0,
        'locusSize': _SET['locusSize'],
        'pValueMantissa': _SET['leadPValueMantissa'],
        'pValueExponent': _SET['leadPValueExponent'],
        'beta': _SET['leadBeta'],
        'fineMappingMethod': _SET['fineMappingMethod'],
        'confidence': _SET['confidence'],
        'isTransQtl': 0,
        'leadVariantId': _SET['leadVariantId'],
        'leadChromosome': _SET['chromosome'],
        'leadPosition': _SET['leadPosition'],
        'leadReferenceAllele': 'C',
        'leadAlternateAllele': 'T',
        'studyId': _SET['studyId'],
        'condition': None,
        'studyTargetId': None,
        'studyTargetApprovedSymbol': None,
        'biosampleId': None,
        'biosampleName': None,
        'traitFromSource': Study.example['traitFromSource'],
        'diseases': [{'id': 'EFO_0021023', 'name': 'isobutyrylcarnitine measurement'}],
        'diseaseTherapeuticAreas': [{'id': 'EFO_0001444', 'name': 'measurement'}],
        'topL2GGeneId': None,
        'topL2GApprovedSymbol': None,
        'topL2GScore': None,
    }

    studyType: StudyType
    variantId: VariantId
    credibleSetId: CredibleSetId
    posteriorProbability: PosteriorProbability
    locusSize: LocusSize
    pValueMantissa: PValueMantissa
    pValueExponent: PValueExponent
    beta: Beta
    fineMappingMethod: FineMappingMethod
    confidence: Confidence
    isTransQtl: IsTransQtl
    leadVariantId: LeadVariantId
    leadChromosome: LeadChromosome
    leadPosition: LeadPosition
    leadReferenceAllele: LeadAllele
    leadAlternateAllele: LeadAllele
    studyId: StudyId
    condition: str | None = Field(min_length=1, description=Study.model_fields['condition'].description)
    studyTargetId: StudyTargetId
    studyTargetApprovedSymbol: ApprovedSymbol
    biosampleId: BiosampleId
    biosampleName: Annotated[str, ClickHouseType('LowCardinality(String)')] | None = Field(
        min_length=1,
        description=f'{Biosample.model_fields["biosampleName"].description}; null when the biosample is not known',
    )
    traitFromSource: Annotated[str, ClickHouseType('LowCardinality(String)')] = Field(
        description=f"{Study.model_fields['traitFromSource'].description}; '' when the study is not known"
    )
    diseases: list[DiseaseName] = Field(
        description='Diseases of the study, in its diseaseIds order, repeats kept and unknown ids left out'
    )
    diseaseTherapeuticAreas: list[DiseaseName] = Field(
        description='Therapeutic areas of the diseases of the study, distinct, by id then name'
    )
    topL2GGeneId: TopL2GGeneId
    topL2GApprovedSymbol: ApprovedSymbol
    topL2GScore: TopL2GScore
