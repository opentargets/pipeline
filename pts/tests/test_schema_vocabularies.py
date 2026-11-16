"""Model vocabularies that the code defines are the code's, in full.

A model lists the values of a closed vocabulary itself, so models do not import transformers. Where the
vocabulary is an enum in the code -- quality flags, study types, clinical stages -- this test keeps the model's
list equal to it: a value added to the enum fails here, and the model is updated in the same change, rather
than the pipeline failing on the first row that carries it.
"""

from enum import Enum

import pytest

from pts.schemas.credible_set import Chromosome, FinemappingMethod
from pts.schemas.dataset.discovery import discover_datasets
from pts.schemas.dataset.spec import dataset_spec, walk
from pts.schemas.study import StudyType
from pts.transformers.clinical_mira.moa.schemas import ActionType
from pts.transformers.clinical_mira.schemas import ClinicalProvider, ClinicalReportType, ClinicalStageCategory
from pts.transformers.clinical_report import ClinicalReportFlags
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.enhancer_to_gene import QualityFlag as EnhancerToGeneFlag
from pts.transformers.study import QualityFlag as StudyFlag

#: (dataset, value path, the enum the code writes it from)
VOCABULARIES: list[tuple[str, str, type[Enum]]] = [
    ('chembl_bioactivity', 'assayActionType', ActionType),
    ('chembl_bioactivity', 'actionTypes[]', ActionType),
    ('clinical_indication', 'maxClinicalStage', ClinicalStageCategory),
    ('clinical_report', 'clinicalStage', ClinicalStageCategory),
    ('clinical_target', 'maxClinicalStage', ClinicalStageCategory),
    ('drug_molecule', 'maximumClinicalStage', ClinicalStageCategory),
    ('evidence_clinical_precedence', 'clinicalStage', ClinicalStageCategory),
    ('clinical_report', 'type', ClinicalReportType),
    ('clinical_report', 'provider', ClinicalProvider),
    ('clinical_report', 'qualityControls[]', ClinicalReportFlags),
    ('colocalisation', 'rightStudyType', StudyType),
    ('credible_set', 'studyType', StudyType),
    ('study', 'studyType', StudyType),
    ('colocalisation', 'chromosome', Chromosome),
    ('credible_set', 'chromosome', Chromosome),
    ('variant', 'chromosome', Chromosome),
    ('credible_set', 'fineMappingMethod', FinemappingMethod),
    ('credible_set', 'qualityControls[]', CredibleSetFlag),
    ('credible_set_variant', 'qualityControls[]', CredibleSetVariantFlag),
    ('enhancer_to_gene', 'qualityControls[]', EnhancerToGeneFlag),
    ('study', 'qualityControls[]', StudyFlag),
]

DATASETS = discover_datasets()


@pytest.mark.parametrize(('dataset', 'path', 'vocabulary'), VOCABULARIES, ids=[f'{d}:{p}' for d, p, _ in VOCABULARIES])
def test_the_model_lists_the_code_vocabulary_in_full_and_in_order(
    dataset: str, path: str, vocabulary: type[Enum]
) -> None:
    nodes = dict(walk(dataset_spec(DATASETS[dataset])))
    assert path in nodes, f'{dataset} has no value at {path}'
    expected = tuple(member.value for member in vocabulary)

    assert nodes[path].constraints.allowed == expected, (
        f'{dataset}:{path} must allow exactly {vocabulary.__module__}.{vocabulary.__name__}, in its order: '
        f'update the Literal in pts/src/pts/schemas/outputs/{dataset}.py to {list(expected)}'
    )
