"""Tests for GWAS credible set evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.gwas_credible_sets import build_raw_evidence

STUDY_SCHEMA = {
    'studyId': pl.String,
    'diseaseIds': pl.List(pl.String),
    'pubmedId': pl.String,
    'publicationDate': pl.String,
}


def build(predictions: dict, credible_set: dict, studies: dict, threshold: float = 0.05) -> pl.DataFrame:
    return build_raw_evidence(
        pl.LazyFrame(predictions),
        pl.LazyFrame(credible_set),
        pl.LazyFrame(studies, schema=STUDY_SCHEMA),
        threshold,
    ).collect()


def by_target_and_disease(df: pl.DataFrame) -> dict:
    return {(r['targetFromSourceId'], r['diseaseFromSourceMappedId']): r for r in df.to_dicts()}


ONE_LOCUS = {'studyLocusId': ['SL1'], 'studyId': ['GCST1']}


def one_study(**overrides: list) -> dict:
    return {'studyId': ['GCST1'], 'diseaseIds': [['EFO_1']], 'pubmedId': [None], 'publicationDate': [None]} | overrides


class TestBuildRawEvidence:
    def test_keeps_predictions_at_or_above_the_threshold(self) -> None:
        result = build(
            {'studyLocusId': ['SL1'] * 3, 'geneId': ['ENSG1', 'ENSG2', 'ENSG3'], 'score': [0.5, 0.05, 0.01]},
            ONE_LOCUS,
            one_study(),
        )
        assert sorted(result['targetFromSourceId'].to_list()) == ['ENSG1', 'ENSG2']

    def test_one_row_per_study_disease(self) -> None:
        result = build(
            {'studyLocusId': ['SL1'], 'geneId': ['ENSG1'], 'score': [0.5]},
            ONE_LOCUS,
            one_study(diseaseIds=[['EFO_1', 'EFO_2']]),
        )
        assert sorted(result['diseaseFromSourceMappedId'].to_list()) == ['EFO_1', 'EFO_2']
        assert result['resourceScore'].to_list() == [0.5, 0.5]

    def test_drops_studies_without_diseases(self) -> None:
        result = build(
            {'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]},
            {'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']},
            {
                'studyId': ['GCST1', 'GCST2'],
                'diseaseIds': [None, []],
                'pubmedId': [None, None],
                'publicationDate': [None, None],
            },
        )
        assert result.height == 0

    def test_drops_unmatched_loci_and_studies(self) -> None:
        result = build(
            {'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]},
            {'studyLocusId': ['SL2'], 'studyId': ['GCST_OTHER']},
            one_study(),
        )
        assert result.height == 0

    def test_curation_date_needs_a_date_shaped_publication_date(self) -> None:
        result = build(
            {'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]},
            {'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']},
            {
                'studyId': ['GCST1', 'GCST2'],
                'diseaseIds': [['EFO_1'], ['EFO_1']],
                'pubmedId': [None, None],
                'publicationDate': ['2020-01-01', 'not-a-date'],
            },
        )
        rows = by_target_and_disease(result)
        assert rows['ENSG1', 'EFO_1']['curationDate'] == '2020-01-01'
        assert rows['ENSG2', 'EFO_1']['curationDate'] is None

    def test_literature_wraps_the_pubmed_id(self) -> None:
        result = build(
            {'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]},
            {'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']},
            {
                'studyId': ['GCST1', 'GCST2'],
                'diseaseIds': [['EFO_1'], ['EFO_1']],
                'pubmedId': ['12345', None],
                'publicationDate': [None, None],
            },
        )
        rows = by_target_and_disease(result)
        assert rows['ENSG1', 'EFO_1']['literature'] == ['12345']
        assert rows['ENSG2', 'EFO_1']['literature'] is None

    def test_datatype_and_datasource(self) -> None:
        result = build({'studyLocusId': ['SL1'], 'geneId': ['ENSG1'], 'score': [0.5]}, ONE_LOCUS, one_study())
        assert result.select('datatypeId', 'datasourceId').row(0) == ('genetic_association', 'gwas_credible_sets')


