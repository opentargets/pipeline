"""Tests for pts.transformers.gwas_evidence, which builds raw GWAS credible-set evidence.

Only `_build_raw_evidence` (pure, no I/O) is tested directly, matching the convention in
`test_drug_mechanism_of_action.py`: the top-level `gwas_evidence` entrypoint is orchestration
(reads, delegates, writes) and is not unit tested here.
"""

import polars as pl

from pts.transformers.evidence.gwas_evidence import _build_raw_evidence


def rows_by_target_and_disease(df: pl.DataFrame) -> dict:
    return {(r['targetFromSourceId'], r['diseaseFromSourceMappedId']): r for r in df.to_dicts()}


#: `pubmedId`/`publicationDate` are all-null in several fixtures below; without an explicit
#: schema polars infers dtype `Null` for an all-`None` column, which the `.str` accessor rejects.
STUDY_INDEX_SCHEMA = {
    'studyId': pl.String,
    'diseaseIds': pl.List(pl.String),
    'pubmedId': pl.String,
    'publicationDate': pl.String,
}


class TestBuildRawEvidence:
    def test_below_threshold_predictions_are_dropped(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.01]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']})
        study_index = pl.DataFrame(
            {
                'studyId': ['GCST1', 'GCST2'],
                'diseaseIds': [['EFO_1'], ['EFO_2']],
                'pubmedId': [None, None],
                'publicationDate': [None, None],
            },
            schema=STUDY_INDEX_SCHEMA,
        )

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)

        assert result.height == 1
        assert result.to_dicts()[0]['targetFromSourceId'] == 'ENSG1'

    def test_one_row_per_disease(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1'], 'geneId': ['ENSG1'], 'score': [0.5]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL1'], 'studyId': ['GCST1']})
        study_index = pl.DataFrame(
            {
                'studyId': ['GCST1'],
                'diseaseIds': [['EFO_1', 'EFO_2']],
                'pubmedId': [None],
                'publicationDate': [None],
            },
            schema=STUDY_INDEX_SCHEMA,
        )

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)

        assert result.height == 2
        assert set(result['diseaseFromSourceMappedId'].to_list()) == {'EFO_1', 'EFO_2'}
        assert set(result['resourceScore'].to_list()) == {0.5}

    def test_null_or_empty_disease_ids_drop_the_row(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']})
        study_index = pl.DataFrame(
            {
                'studyId': ['GCST1', 'GCST2'],
                'diseaseIds': [None, []],
                'pubmedId': [None, None],
                'publicationDate': [None, None],
            },
            schema={
                'studyId': pl.String,
                'diseaseIds': pl.List(pl.String),
                'pubmedId': pl.String,
                'publicationDate': pl.String,
            },
        )

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)

        assert result.height == 0

    def test_unmatched_study_locus_or_study_is_dropped(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1'], 'geneId': ['ENSG1'], 'score': [0.5]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL_OTHER'], 'studyId': ['GCST1']})
        study_index = pl.DataFrame(
            {
                'studyId': ['GCST1'],
                'diseaseIds': [['EFO_1']],
                'pubmedId': [None],
                'publicationDate': [None],
            },
            schema=STUDY_INDEX_SCHEMA,
        )

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)

        assert result.height == 0

    def test_curation_date_requires_iso_date_shaped_publication_date(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']})
        study_index = pl.DataFrame({
            'studyId': ['GCST1', 'GCST2'],
            'diseaseIds': [['EFO_1'], ['EFO_1']],
            'pubmedId': [None, None],
            'publicationDate': ['2020-01-01', 'not-a-date'],
        })

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)
        by_target = rows_by_target_and_disease(result)

        assert by_target[('ENSG1', 'EFO_1')]['curationDate'] == '2020-01-01'
        assert by_target[('ENSG2', 'EFO_1')]['curationDate'] is None

    def test_literature_wraps_pubmed_id_in_a_single_element_list(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'geneId': ['ENSG1', 'ENSG2'], 'score': [0.5, 0.5]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL1', 'SL2'], 'studyId': ['GCST1', 'GCST2']})
        study_index = pl.DataFrame(
            {
                'studyId': ['GCST1', 'GCST2'],
                'diseaseIds': [['EFO_1'], ['EFO_1']],
                'pubmedId': ['12345', None],
                'publicationDate': [None, None],
            },
            schema=STUDY_INDEX_SCHEMA,
        )

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)
        by_target = rows_by_target_and_disease(result)

        assert by_target[('ENSG1', 'EFO_1')]['literature'] == ['12345']
        assert by_target[('ENSG2', 'EFO_1')]['literature'] is None

    def test_literal_datatype_and_datasource(self) -> None:
        predictions = pl.DataFrame({'studyLocusId': ['SL1'], 'geneId': ['ENSG1'], 'score': [0.5]})
        credible_set = pl.DataFrame({'studyLocusId': ['SL1'], 'studyId': ['GCST1']})
        study_index = pl.DataFrame(
            {
                'studyId': ['GCST1'],
                'diseaseIds': [['EFO_1']],
                'pubmedId': [None],
                'publicationDate': [None],
            },
            schema=STUDY_INDEX_SCHEMA,
        )

        result = _build_raw_evidence(predictions, credible_set, study_index, threshold=0.05)
        row = result.to_dicts()[0]

        assert row['datatypeId'] == 'genetic_association'
        assert row['datasourceId'] == 'gwas_credible_sets'
