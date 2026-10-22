"""Tests for the evidence runner and its raw evidence check."""

from __future__ import annotations

from pathlib import Path

import pandera.errors as pe
import polars as pl
import pytest

from pts.schemas.outputs.gwas_evidence import GwasCredibleSetEvidence
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.evidence import QC_COLUMN, EvidenceFlags
from pts.transformers.evidence.utils.luts import ValidationLuts
from pts.transformers.evidence.utils.run import run_evidence, write_evidence
from pts.transformers.utils.dataset import sink_dataset
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

    def test_postprocess_checks_the_raw_evidence(self) -> None:
        spec = EvidenceDatasource(datasource_id='eva', unique_fields=['studyLocusId'], score=pl.lit(1.0))
        luts = ValidationLuts(
            disease=pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']}),
            target=pl.DataFrame({'targetFromSourceId': ['ENSG1'], 'targetId': ['ENSG1'], 'biotype': ['pc']}),
        )
        with pytest.raises(ValueError, match='studyLocusId'):
            spec.postprocess(raw(), luts)


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
    with pytest.raises(ValueError, match=str(tmp_path / 'out' / 'valid')):
        run_evidence(raw(), SPEC, **kwargs)


def test_valid_and_failed_evidence_are_computed_in_one_pass(tmp_path: Path) -> None:
    batches: list[int] = []

    def count(s: pl.Series) -> pl.Series:
        batches.append(len(s))
        return s

    counted = raw().with_columns(pl.col('resourceScore').map_batches(count, return_dtype=pl.Float64))
    dst = tmp_path / 'out'
    run_evidence(
        counted,
        SPEC,
        disease=write_disease(tmp_path / 'disease', ['EFO_1']),
        target=write_target(tmp_path / 'target', ['ENSG1']),
        destination={'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
    )

    assert sum(batches) == 2
    assert pl.read_parquet(dst / 'valid')['targetId'].to_list() == ['ENSG1']
    assert pl.read_parquet(dst / 'failed')['targetFromSourceId'].to_list() == ['NOT_A_TARGET']


def test_an_existing_failed_destination_stops_both_writes(tmp_path: Path) -> None:
    (tmp_path / 'out' / 'failed').mkdir(parents=True)
    dst = {'evidence': str(tmp_path / 'out' / 'valid'), 'failed_evidence': str(tmp_path / 'out' / 'failed')}
    with pytest.raises(ValueError, match='failed'):
        run_evidence(
            raw(),
            SPEC,
            disease=write_disease(tmp_path / 'disease', ['EFO_1']),
            target=write_target(tmp_path / 'target', ['ENSG1']),
            destination=dst,
        )
    assert not (tmp_path / 'out' / 'valid').exists()


def test_evidence_violating_its_schema_is_not_written_but_failed_evidence_is(tmp_path: Path) -> None:
    spec = EvidenceDatasource(
        datasource_id='eva',
        unique_fields=['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId'],
        score=pl.col('resourceScore'),
        # eva evidence lacks the GWAS columns, and its ids are not Ensembl-shaped
        schema=GwasCredibleSetEvidence,
    )
    dst = tmp_path / 'out'
    with pytest.raises(pe.SchemaErrors):
        run_evidence(
            raw(),
            spec,
            disease=write_disease(tmp_path / 'disease', ['EFO_1']),
            target=write_target(tmp_path / 'target', ['ENSG1']),
            destination={'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        )
    assert not (dst / 'valid').exists()
    assert pl.read_parquet(dst / 'failed')['targetFromSourceId'].to_list() == ['NOT_A_TARGET']


def test_validated_evidence_and_alongside_sinks_are_computed_in_one_pass(tmp_path: Path) -> None:
    batches: list[int] = []

    def count(s: pl.Series) -> pl.Series:
        batches.append(len(s))
        return s

    counted = raw().with_columns(pl.col('resourceScore').map_batches(count, return_dtype=pl.Float64))
    luts = ValidationLuts(
        disease=pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']}),
        target=pl.DataFrame({'targetFromSourceId': ['ENSG1'], 'targetId': ['ENSG1'], 'biotype': ['pc']}),
    )
    evidence = SPEC.postprocess(counted, luts)
    dst = tmp_path / 'out'
    with pytest.raises(pe.SchemaErrors):
        write_evidence(
            evidence,
            {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
            alongside=[sink_dataset(evidence.lf.select('id'), str(dst / 'ids'))],
            schema=GwasCredibleSetEvidence,
        )

    assert sum(batches) == 2
    assert pl.read_parquet(dst / 'ids').height == 2
