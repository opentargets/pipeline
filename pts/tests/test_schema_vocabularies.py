"""Model vocabularies that the code defines are the code's, in full.

A model lists the values of a closed vocabulary itself, so models do not import transformers. Where the
vocabulary is an enum in the code -- quality flags, study types, clinical stages -- this test keeps the model's
list equal to it: a value added to the enum fails here, and the model is updated in the same change, rather
than the pipeline failing on the first row that carries it. Where a transformer writes values from a plain
constant, the model's list must include every one of them.
"""

from collections.abc import Iterable
from enum import Enum

import pytest
from ot_datamodels.discovery import discover_datasets
from ot_datamodels.spec import dataset_spec, walk

from pts.schemas.credible_set import CHROMOSOMES, FINEMAPPING_METHODS
from pts.schemas.study import STUDY_TYPES
from pts.transformers.clinical_mira.moa.schemas import ActionType
from pts.transformers.clinical_mira.schemas import ClinicalProvider, ClinicalReportType, ClinicalStageCategory
from pts.transformers.clinical_report import ClinicalReportFlags
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.enhancer_to_gene import QualityFlag as EnhancerToGeneFlag
from pts.transformers.evidence import europepmc, gene2phenotype
from pts.transformers.evidence.gene_burden import bipex, brava, schema
from pts.transformers.lof_curation import METHOD as LOF_CURATION_METHOD
from pts.transformers.study import QualityFlag as StudyFlag
from pts.transformers.variant.annotators.gpn_star import METHOD as GPN_STAR_METHOD

#: (dataset, value path, the enum or vocabulary the code writes it from)
VOCABULARIES: list[tuple[str, str, type[Enum] | tuple[str, ...]]] = [
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
    ('colocalisation', 'rightStudyType', STUDY_TYPES),
    ('credible_set', 'studyType', STUDY_TYPES),
    ('study', 'studyType', STUDY_TYPES),
    ('colocalisation', 'chromosome', CHROMOSOMES),
    ('credible_set', 'chromosome', CHROMOSOMES),
    ('variant', 'chromosome', CHROMOSOMES),
    ('credible_set', 'fineMappingMethod', FINEMAPPING_METHODS),
    ('credible_set', 'qualityControls[]', CredibleSetFlag),
    ('credible_set_variant', 'qualityControls[]', CredibleSetVariantFlag),
    ('enhancer_to_gene', 'qualityControls[]', EnhancerToGeneFlag),
    ('study', 'qualityControls[]', StudyFlag),
]

#: (dataset, value path, values a transformer writes from a constant)
WRITTEN: list[tuple[str, str, Iterable[str]]] = [
    ('evidence_europepmc', 'textMiningSentences[]/section', europepmc.SECTIONS_OF_INTEREST),
    ('evidence_gene2phenotype', 'confidence', gene2phenotype.CONFIDENCE_SCORES),
    ('evidence_gene_burden', 'ancestry', brava.ANCESTRY_LABELS.values()),
    ('evidence_gene_burden', 'urls[]/niceName', [f'{s.STUDY.project_id} browser' for s in (bipex, schema)]),
    ('variant', 'variantEffect[]/method', [GPN_STAR_METHOD, LOF_CURATION_METHOD]),
]

DATASETS = discover_datasets()


@pytest.mark.parametrize(('dataset', 'path', 'vocabulary'), VOCABULARIES, ids=[f'{d}:{p}' for d, p, _ in VOCABULARIES])
def test_the_model_lists_the_code_vocabulary_in_full_and_in_order(
    dataset: str, path: str, vocabulary: type[Enum] | tuple[str, ...]
) -> None:
    nodes = dict(walk(dataset_spec(DATASETS[dataset])))
    assert path in nodes, f'{dataset} has no value at {path}'
    expected = vocabulary if isinstance(vocabulary, tuple) else tuple(member.value for member in vocabulary)
    name = 'the vocabulary' if isinstance(vocabulary, tuple) else f'{vocabulary.__module__}.{vocabulary.__name__}'

    assert nodes[path].constraints.allowed == expected, (
        f'{dataset}:{path} must allow exactly {name}, in its order: '
        f'update the Literal in datamodels/src/ot_datamodels/outputs/{dataset}.py to {list(expected)}'
    )


@pytest.mark.parametrize(('dataset', 'path', 'values'), WRITTEN, ids=[f'{d}:{p}' for d, p, _ in WRITTEN])
def test_the_model_allows_every_value_the_code_writes(dataset: str, path: str, values: Iterable[str]) -> None:
    nodes = dict(walk(dataset_spec(DATASETS[dataset])))
    assert path in nodes, f'{dataset} has no value at {path}'
    allowed = nodes[path].constraints.allowed
    assert allowed is not None, f'{dataset}:{path} is not a closed vocabulary'

    missing = sorted(set(values) - set(allowed))
    assert not missing, f'add {missing} to the Literal at {path} in datamodels/src/ot_datamodels/outputs/{dataset}.py'
