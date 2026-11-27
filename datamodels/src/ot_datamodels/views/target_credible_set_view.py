"""Target credible set: the `target_credible_set_view` view (pts `credible_set_view`).

One row per target and credible set of a study of that target (`study.geneId`), for the target page's molQTL
table: the set's lead variant and statistics, and its study's condition and biosample.
"""

from typing import ClassVar

from pydantic import Field

from ot_datamodels import DatasetModel, from_output
from ot_datamodels.outputs.biosample import Biosample
from ot_datamodels.outputs.credible_set import CredibleSet
from ot_datamodels.outputs.credible_set_variant import CredibleSetVariant
from ot_datamodels.outputs.study import Study
from ot_datamodels.outputs.target import Target
from ot_datamodels.serving import Serving
from ot_datamodels.views.credible_set_view import CredibleSetView, LeadAllele

TargetId = from_output(Target, 'id')
CredibleSetId = from_output(CredibleSet, 'id')
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
StudyType = from_output(CredibleSet, 'studyType')
Condition = from_output(Study, 'condition')
BiosampleId = from_output(Study, 'biosampleFromSourceId')
LocusSize = from_output(CredibleSetView, 'locusSize')

_SET = CredibleSetView.example


class TargetCredibleSetView(DatasetModel):
    """Target credible set."""

    dataset_name: ClassVar[str] = 'target_credible_set_view'
    dataset_title: ClassVar[str] = 'Target credible set'
    dataset_description: ClassVar[str] = 'The credible sets of the studies of each target'
    dataset_tags: ClassVar[tuple[str, ...]] = ('Genetics',)
    clickhouse: ClassVar[Serving] = Serving(
        engine='mergetree',
        order_by=('targetId', 'pValueExponent', 'pValueMantissa', 'credibleSetId'),
        column_types={
            'confidence': 'UInt8',
            'condition': 'LowCardinality(String)',
            'biosampleName': 'LowCardinality(String)',
        },
    )
    example: ClassVar[dict] = {
        'targetId': 'ENSG00000198931',
        'credibleSetId': _SET['credibleSetId'],
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
        'studyId': 'QTD000001',
        'studyType': 'eqtl',
        'condition': 'naive',
        'biosampleId': 'UBERON_0001969',
        'biosampleName': 'blood plasma',
        'locusSize': _SET['locusSize'],
    }

    targetId: TargetId
    credibleSetId: CredibleSetId
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
    studyType: StudyType
    condition: Condition
    biosampleId: BiosampleId
    biosampleName: str = Field(
        description=f"{Biosample.model_fields['biosampleName'].description}; '' when the biosample is not known"
    )
    locusSize: LocusSize
