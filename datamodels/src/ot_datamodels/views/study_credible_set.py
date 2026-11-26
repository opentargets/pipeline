"""Study credible set: the `study_credible_set` view (pts `credible_set_view`), served as `study_credible_sets`.

One row per GWAS credible set, for the study page: its lead variant and statistics, and its top L2G gene (the
highest score, then the lowest `geneId`; null without predictions).
"""

from typing import ClassVar

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.credible_set import CredibleSet
from ot_datamodels.outputs.credible_set_variant import CredibleSetVariant
from ot_datamodels.outputs.l2g_prediction import L2gPrediction
from ot_datamodels.views.credible_set_view import ApprovedSymbol, CredibleSetView, LeadAllele

StudyId = from_output(CredibleSet, 'studyId')
CredibleSetId = from_output(CredibleSet, 'id')
LeadVariantId = from_output(CredibleSet, 'leadVariantId')
LeadChromosome = from_output(CredibleSet, 'chromosome')
LeadPosition = from_output(CredibleSetView, 'leadPosition')
PValueMantissa = from_output(CredibleSetVariant, 'pValueMantissa')
PValueExponent = from_output(CredibleSetVariant, 'pValueExponent')
Beta = from_output(CredibleSetVariant, 'reportedBeta')
FineMappingMethod = from_output(CredibleSet, 'fineMappingMethod')
Confidence = from_output(CredibleSet, 'confidence')
LocusSize = from_output(CredibleSetView, 'locusSize')
TopL2GGeneId = from_output(L2gPrediction, 'geneId', nullable=True)
TopL2GScore = from_output(L2gPrediction, 'score', nullable=True)

_SET = CredibleSetView.example


class StudyCredibleSet(DatasetModel):
    """Study credible set."""

    dataset_name: ClassVar[str] = 'study_credible_set'
    dataset_title: ClassVar[str] = 'Study credible set'
    dataset_description: ClassVar[str] = (
        'The GWAS credible sets of each study, with their lead variant and top L2G gene'
    )
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    example: ClassVar[dict] = {
        'studyId': _SET['studyId'],
        'credibleSetId': _SET['credibleSetId'],
        'leadVariantId': _SET['leadVariantId'],
        'leadChromosome': _SET['chromosome'],
        'leadPosition': _SET['leadPosition'],
        'leadReferenceAllele': 'C',
        'leadAlternateAllele': 'T',
        'pValueMantissa': _SET['leadPValueMantissa'],
        'pValueExponent': _SET['leadPValueExponent'],
        'beta': _SET['leadBeta'],
        'fineMappingMethod': _SET['fineMappingMethod'],
        'confidence': _SET['confidence'],
        'locusSize': _SET['locusSize'],
        'topL2GGeneId': L2gPrediction.example['geneId'],
        'topL2GApprovedSymbol': 'KCNH1',
        'topL2GScore': L2gPrediction.example['score'],
    }

    studyId: StudyId
    credibleSetId: CredibleSetId
    leadVariantId: LeadVariantId
    leadChromosome: LeadChromosome
    leadPosition: LeadPosition
    leadReferenceAllele: LeadAllele
    leadAlternateAllele: LeadAllele
    pValueMantissa: PValueMantissa
    pValueExponent: PValueExponent
    beta: Beta
    fineMappingMethod: FineMappingMethod
    confidence: Confidence
    locusSize: LocusSize
    topL2GGeneId: TopL2GGeneId
    topL2GApprovedSymbol: ApprovedSymbol
    topL2GScore: TopL2GScore
