"""Tests for the evidence runner and its raw evidence check."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.evidence import QC_COLUMN, EvidenceFlags
from pts.transformers.evidence.utils.luts import ValidationLuts
from pts.transformers.evidence.utils.run import run_evidence
from tests.evidence.fixtures import write_disease, write_literature, write_target

SPEC = EvidenceDatasource(
    datasource_id='eva',
    unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId'],
    score=pl.col('resourceScore'),
)


def raw(**columns: list) -> pl.LazyFrame:
    base = {
        'datasourceId': ['eva', 'eva'],
        'targetFromSourceId': ['ENSG1', 'NOT_A_TARGET'],
        'diseaseFromSourceMappedId': ['EFO_1', 'EFO_1'],
        'resourceScore': [0.5, 0.5],
    }
    return pl.LazyFrame(base | columns)


class TestCheckRawEvidence:
    def test_accepts_a_complete_frame(self) -> None:
        SPEC.check_raw(raw())

    def test_rejects_a_missing_required_column(self) -> None:
        with pytest.raises(ValueError, match='diseaseFromSourceMappedId'):
            SPEC.check_raw(raw().drop('diseaseFromSourceMappedId'))

    def test_rejects_a_missing_unique_field(self) -> None:
        spec = EvidenceDatasource(datasource_id='eva', unique_fields=['studyLocusId'], score=pl.lit(1.0))
        with pytest.raises(ValueError, match='studyLocusId'):
            spec.check_raw(raw())

    def test_derived_unique_fields_need_not_be_raw(self) -> None:
        SPEC.check_raw(raw())

    def test_accepts_columns_outside_the_schema(self) -> None:
        SPEC.check_raw(raw(notInSchema=['x', 'y']))


class TestRecipeLiterature:
    def test_runs_without_a_publication_table_when_there_is_no_literature(self) -> None:
        luts = ValidationLuts(
            disease=pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']}),
            target=pl.DataFrame({'targetFromSourceId': ['ENSG1'], 'targetId': ['ENSG1'], 'biotype': ['pc']}),
        )
        assert SPEC.postprocess(raw(), luts).valid().collect().height == 1

    def test_refuses_literature_without_a_publication_table(self) -> None:
        luts = ValidationLuts(disease=pl.DataFrame(), target=pl.DataFrame())
        with pytest.raises(ValueError, match='no publication look-up table'):
            SPEC.postprocess(raw(literature=[['1'], None]), luts)


class TestRunEvidence:
    def test_writes_valid_and_failed_evidence(self, tmp_path: Path) -> None:
        dst = tmp_path / 'out'
        run_evidence(
            raw(),
            SPEC,
            disease=write_disease(tmp_path / 'disease', ['EFO_1']),
            target=write_target(tmp_path / 'target', ['ENSG1']),
            destination={'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        )

        valid = pl.read_parquet(dst / 'valid')
        failed = pl.read_parquet(dst / 'failed')
        assert valid['targetId'].to_list() == ['ENSG1']
        assert failed['targetFromSourceId'].to_list() == ['NOT_A_TARGET']
        assert EvidenceFlags.INVALID_TARGET in failed[QC_COLUMN].item().to_list()

    def test_dates_evidence_from_literature(self, tmp_path: Path) -> None:
        dst = tmp_path / 'out'
        run_evidence(
            raw(literature=[['1'], ['2']]),
            SPEC,
            disease=write_disease(tmp_path / 'disease', ['EFO_1']),
            target=write_target(tmp_path / 'target', ['ENSG1']),
            literature=write_literature(tmp_path / 'literature', {'1': '2001-01-01'}),
            destination={'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        )

        valid = pl.read_parquet(dst / 'valid')
        assert valid.select('publicationDate', 'evidenceDate').row(0) == ('2001-01-01', '2001-01-01')

    def test_needs_the_literature_export_for_evidence_with_literature(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match='no literature export'):
            run_evidence(
                raw(literature=[['1'], None]),
                SPEC,
                disease=str(tmp_path),
                target=str(tmp_path),
                destination={},
            )


def test_rerun_into_an_existing_destination_fails_and_names_it(tmp_path: Path) -> None:
    dst = {'evidence': str(tmp_path / 'out' / 'valid'), 'failed_evidence': str(tmp_path / 'out' / 'failed')}
    kwargs = {
        'disease': write_disease(tmp_path / 'disease', ['EFO_1']),
        'target': write_target(tmp_path / 'target', ['ENSG1']),
        'destination': dst,
    }
    run_evidence(raw(), SPEC, **kwargs)
    with pytest.raises(ValueError, match='valid'):
        run_evidence(raw(), SPEC, **kwargs)
