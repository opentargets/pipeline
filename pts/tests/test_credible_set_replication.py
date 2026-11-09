"""The REPLICATED rule: credible sets whose lead is reported on one of their traits by two independent reports."""

from typing import Any

import polars as pl
import pytest

from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.replication import replicated_sets
from pts.transformers.disease.coalescing import MEASUREMENT


def _sets(*rows: dict[str, Any]) -> pl.LazyFrame:
    defaults = {'qualityControls': [CredibleSetFlag.TOP_HIT.value]}
    return pl.LazyFrame(
        [{**defaults, **row} for row in rows],
        schema={
            'id': pl.String,
            'studyId': pl.String,
            'leadVariantId': pl.String,
            'qualityControls': pl.List(pl.String),
        },
    )


def _ids(frame: pl.LazyFrame) -> set[str]:
    ids = frame.collect()
    assert ids.columns == ['id']
    return set(ids['id'])


DISEASES = pl.LazyFrame({
    'id': ['EFO_disease', 'EFO_other', 'EFO_bmi', 'EFO_height'],
    'therapeuticAreas': [['EFO_area'], ['EFO_area'], [MEASUREMENT], [MEASUREMENT]],
})
LEAD = '1_100_A_T'


def _gwas(study_id: str, diseases: list[str], pubmed: str | None = None, **values: Any) -> dict[str, Any]:
    return {
        'studyId': study_id,
        'studyType': 'gwas',
        'diseaseIds': diseases,
        'pubmedId': pubmed or study_id,
        'cohorts': None,
        'ldPopulationStructure': None,
        'geneId': None,
        **values,
    }


def _replicated(
    studies: list[dict[str, Any]], sets: list[tuple[str, str, str]] | None = None, disease: pl.LazyFrame = DISEASES
) -> set[str]:
    """The ids of the replicated sets of (id, studyId, leadVariantId), by default one set per study on LEAD."""
    sets = sets or [(f'cs_{study["studyId"]}', study['studyId'], LEAD) for study in studies]
    frame = _sets(*({'id': i, 'studyId': s, 'leadVariantId': v} for i, s, v in sets))
    study_frame = pl.LazyFrame(
        studies,
        schema={
            'studyId': pl.String,
            'studyType': pl.String,
            'diseaseIds': pl.List(pl.String),
            'pubmedId': pl.String,
            'cohorts': pl.List(pl.String),
            'ldPopulationStructure': pl.List(pl.Struct({'ldPopulation': pl.String, 'relativeSampleSize': pl.Float64})),
            'geneId': pl.String,
        },
    )
    return _ids(replicated_sets(frame, study_frame, disease))


def test_independent_gwas_studies_of_one_disease_and_lead_replicate() -> None:
    assert _replicated([_gwas('a', ['EFO_disease']), _gwas('b', ['EFO_disease'])]) == {'cs_a', 'cs_b'}


def test_gwas_studies_of_one_publication_cohort_and_population_count_once() -> None:
    same = {
        'pubmed': 'p',
        'cohorts': ['UKB'],
        'ldPopulationStructure': [{'ldPopulation': 'nfe', 'relativeSampleSize': 1.0}],
    }
    assert _replicated([_gwas('a', ['EFO_disease'], **same), _gwas('b', ['EFO_disease'], **same)]) == set()


def test_a_gwas_lead_alone_or_on_another_disease_is_not_replicated() -> None:
    assert _replicated([_gwas('a', ['EFO_disease'])]) == set()
    assert _replicated([_gwas('a', ['EFO_disease']), _gwas('b', ['EFO_other'])]) == set()


def test_a_study_of_two_measurements_replicates_with_a_study_sharing_one() -> None:
    assert _replicated([_gwas('a', ['EFO_bmi', 'EFO_height']), _gwas('b', ['EFO_height'])]) == {'cs_a', 'cs_b'}


def test_a_study_of_measurements_and_diseases_replicates_on_its_diseases_only() -> None:
    studies = [_gwas('a', ['EFO_bmi', 'EFO_disease']), _gwas('b', ['EFO_bmi']), _gwas('c', ['EFO_disease'])]
    assert _replicated(studies) == {'cs_a', 'cs_c'}


def test_a_term_absent_from_the_disease_index_is_a_disease() -> None:
    assert _replicated([_gwas('a', ['EFO_new']), _gwas('b', ['EFO_new'])]) == {'cs_a', 'cs_b'}


def test_two_qtl_sets_with_one_lead_and_gene_replicate_even_in_one_study() -> None:
    qtl = {'studyType': 'eqtl', 'diseaseIds': [], 'geneId': 'g'}
    studies = [_gwas('q1', [], **qtl), _gwas('q2', [], **qtl)]
    across = [('x', 'q1', LEAD), ('y', 'q2', LEAD), ('z', 'q2', '1_999_A_T')]
    assert _replicated(studies, across) == {'x', 'y'}
    within = [('x', 'q1', LEAD), ('y', 'q1', LEAD)]
    assert _replicated(studies, within) == {'x', 'y'}


TRAIT_TERMS = pl.LazyFrame({
    'id': ['bmi', 'height', 'ibd', 'ra'],
    'therapeuticAreas': [[MEASUREMENT], [MEASUREMENT], ['EFO_area'], ['EFO_area']],
})
"""Two measurements, `bmi` and `height`, and two diseases, `ibd` and `ra`."""


@pytest.mark.parametrize(
    ('disease_ids', 'traits'),
    [
        (['bmi', 'ibd'], [['ibd']]),
        (['ibd'], [['ibd']]),
        (['ra', 'ibd'], [['ibd', 'ra']]),
        (['height', 'bmi'], [['bmi'], ['height']]),
        (['ibd', 'bmi', 'height'], [['ibd']]),
        (['ra', 'ibd', 'bmi'], [['ibd', 'ra']]),
        (['ra', 'bmi', 'ibd', 'height'], [['ibd', 'ra']]),
        ([], []),
    ],
    ids=[
        'one measurement, one disease',
        'one disease',
        'many diseases',
        'many measurements',
        'one disease, many measurements',
        'many diseases, one measurement',
        'many diseases, many measurements',
        'empty list',
    ],
)
def test_the_traits_of_a_gwas_study_depend_on_the_kind_of_its_terms(
    disease_ids: list[str], traits: list[list[str]]
) -> None:
    """A study replicates another exactly on their shared traits, so a study of one trait tells whether it is one."""
    for trait in traits:
        assert _replicated([_gwas('s', disease_ids), _gwas('t', trait)], disease=TRAIT_TERMS) == {'cs_s', 'cs_t'}, trait
    for term in ['bmi', 'height', 'ibd', 'ra']:
        replicated = _replicated([_gwas('s', disease_ids), _gwas('t', [term])], disease=TRAIT_TERMS)
        assert bool(replicated) is ([term] in traits), term
