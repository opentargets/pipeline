"""Annotation of the validated credible sets: the TRANS_QTL and REPLICATED flags."""

from typing import Any

import polars as pl
import pytest

from pts.transformers.credible_set.annotation import MEASUREMENT_THERAPEUTIC_AREA, flag_replicated, flag_trans_qtl
from pts.transformers.credible_set.flags import CredibleSetFlag

TSS = 10_000_000
"""The TSS of gene `g`, on chromosome 1."""
TARGETS = pl.LazyFrame({
    'id': ['g'],
    'tss': [TSS],
    'genomicLocation': [{'chromosome': '1', 'start': TSS, 'end': TSS + 1, 'strand': 1}],
})


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


def _flags(frame: pl.LazyFrame) -> dict[str, set[str]]:
    rows = frame.collect().iter_rows(named=True)
    return {row['id']: set(row['qualityControls']) - {CredibleSetFlag.TOP_HIT} for row in rows}


# TRANS_QTL


def _qtl_studies(**rows: tuple[str, str | None]) -> pl.LazyFrame:
    """Studies of `studyId=(studyType, geneId)` keyword pairs."""
    return pl.LazyFrame(
        [{'studyId': study, 'studyType': kind, 'geneId': gene} for study, (kind, gene) in rows.items()],
        schema={'studyId': pl.String, 'studyType': pl.String, 'geneId': pl.String},
    )


def _trans(
    variant_ids: list[str], study_type: str = 'pqtl', gene: str | None = 'g', threshold: int = 5_000_000
) -> bool:
    sets = _sets({'id': 'cs', 'studyId': 's', 'leadVariantId': variant_ids[0]})
    variants = pl.LazyFrame({'credibleSetId': ['cs'] * len(variant_ids), 'variantId': variant_ids})
    flagged = flag_trans_qtl(sets, variants, _qtl_studies(s=(study_type, gene)), TARGETS, threshold)
    return CredibleSetFlag.TRANS_QTL in _flags(flagged)['cs']


FAR = [f'1_{TSS + 5_000_001}_A_T', f'1_{TSS - 6_000_000}_C_G']


def test_a_qtl_with_every_variant_beyond_the_threshold_is_trans() -> None:
    assert _trans(FAR)


def test_a_qtl_with_one_variant_within_the_threshold_is_not_trans() -> None:
    assert not _trans([*FAR, f'1_{TSS + 5_000_000}_A_T'])


def test_a_qtl_on_another_chromosome_than_its_gene_is_trans() -> None:
    assert _trans([f'2_{TSS}_A_T'])


@pytest.mark.parametrize(
    ('study_type', 'gene'),
    [('gwas', 'g'), ('pqtl', None), ('pqtl', 'unknown')],
    ids=['gwas', 'no gene', 'unknown gene'],
)
def test_a_gwas_or_a_qtl_without_a_located_gene_is_never_trans(study_type: str, gene: str | None) -> None:
    assert not _trans([f'2_{TSS}_A_T'], study_type, gene)


def test_the_trans_threshold_is_a_setting() -> None:
    assert _trans([f'1_{TSS + 1_001}_A_T'], threshold=1_000)


def test_trans_flagging_keeps_rows_order_and_columns() -> None:
    sets = _sets(*({'id': f'cs{i}', 'studyId': 's', 'leadVariantId': f'2_{i + 1}_A_T'} for i in range(3)))
    variants = pl.LazyFrame({'credibleSetId': ['cs0', 'cs1', 'cs2'], 'variantId': ['2_1_A_T', '2_2_A_T', '2_3_A_T']})
    flagged = flag_trans_qtl(sets, variants, _qtl_studies(s=('pqtl', 'g')), TARGETS, 5_000_000).collect()
    assert flagged.columns == sets.collect_schema().names()
    assert flagged['id'].to_list() == ['cs0', 'cs1', 'cs2']


# REPLICATED

DISEASES = pl.LazyFrame({
    'id': ['EFO_disease', 'EFO_other', 'EFO_bmi', 'EFO_height'],
    'therapeuticAreas': [['EFO_area'], ['EFO_area'], [MEASUREMENT_THERAPEUTIC_AREA], [MEASUREMENT_THERAPEUTIC_AREA]],
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


def _replicated(studies: list[dict[str, Any]], sets: list[tuple[str, str, str]] | None = None) -> set[str]:
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
    flags = _flags(flag_replicated(frame, study_frame, DISEASES))
    return {set_id for set_id, flagged in flags.items() if CredibleSetFlag.REPLICATED in flagged}


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
