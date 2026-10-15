"""Tests for pts.transformers.evidence.encore_evidence.

Only `_process_encore_evidence` (pure, no I/O) is tested directly, matching the convention in
`test_gwas_evidence.py`: the top-level `encore_evidence` entrypoint is orchestration (reads,
delegates, writes) and is not unit tested here.
"""

import polars as pl

from pts.transformers.evidence.encore_evidence import _process_encore_evidence

UNIQUE_FIELDS = ['targetId', 'targetFromSourceId', 'diseaseId', 'datasourceId', 'interactingTargetFromSourceId']


def _raw_row(**overrides: object) -> dict:
    row = {
        'datasourceId': 'encore',
        'targetFromSourceId': 'CHEK1',
        'diseaseFromSourceMappedId': 'EFO_1',
        'interactingTargetFromSourceId': 'PARK7',
        'geneticInteractionScore': -8.0,
    }
    row.update(overrides)
    return row


def _luts() -> tuple[pl.DataFrame, pl.DataFrame]:
    disease_lut = pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']})
    target_lut = pl.DataFrame({
        'targetFromSourceId': ['CHEK1'],
        'targetId': ['ENSG00000149554'],
        'biotype': ['protein_coding'],
    })
    return disease_lut, target_lut


class TestProcessEncoreEvidence:
    def test_keeps_only_the_encore_datasource(self) -> None:
        df = pl.DataFrame([_raw_row(), _raw_row(datasourceId='other', targetFromSourceId='OTHER')])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, UNIQUE_FIELDS)

        assert result.height == 1
        assert result.to_dicts()[0]['targetFromSourceId'] == 'CHEK1'

    def test_score_is_the_absolute_genetic_interaction_score_over_16(self) -> None:
        """score = ABS(geneticInteractionScore) / 16 -- ABS so a positive (cooperative)
        interaction scores the same magnitude as the equivalent negative (antagonistic) one.
        """
        df = pl.DataFrame([
            _raw_row(geneticInteractionScore=-16.0, interactingTargetFromSourceId='A'),
            _raw_row(geneticInteractionScore=-8.0, interactingTargetFromSourceId='B'),
            _raw_row(geneticInteractionScore=0.0, interactingTargetFromSourceId='C'),
            _raw_row(
                geneticInteractionScore=8.0, interactingTargetFromSourceId='E', geneticInteractionType='cooperative'
            ),
        ])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, UNIQUE_FIELDS)
        by_interactor = {r['interactingTargetFromSourceId']: r['score'] for r in result.to_dicts()}

        assert by_interactor == {'A': 1.0, 'B': 0.5, 'C': 0.0, 'E': 0.5}

    def test_score_beyond_16_in_magnitude_is_flagged_not_clamped(self) -> None:
        df = pl.DataFrame([_raw_row(geneticInteractionScore=-32.0)])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, UNIQUE_FIELDS)
        row = result.to_dicts()[0]

        assert row['score'] == 2.0
        assert row['qualityControls'] == ['No valid score']

    def test_resolves_disease_and_target_and_flags_unresolved(self) -> None:
        df = pl.DataFrame([
            _raw_row(),
            _raw_row(diseaseFromSourceMappedId='EFO_MISSING', interactingTargetFromSourceId='X'),
        ])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, UNIQUE_FIELDS)
        by_interactor = {r['interactingTargetFromSourceId']: r for r in result.to_dicts()}

        assert by_interactor['PARK7']['diseaseId'] == 'EFO_1'
        assert by_interactor['PARK7']['qualityControls'] == []
        assert by_interactor['X']['diseaseId'] is None
        assert by_interactor['X']['qualityControls'] == ['No valid disease']

    def test_evidence_date_falls_back_to_release_date(self) -> None:
        df = pl.DataFrame([_raw_row(releaseDate='2024-01-15')])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, UNIQUE_FIELDS)

        assert result.to_dicts()[0]['evidenceDate'] == '2024-01-15'

    def test_handles_the_disease_cell_lines_struct_list_in_unique_fields(self) -> None:
        """`diseaseCellLines` (list[struct]) is a real production `unique_fields` entry -- must
        not crash identifier assignment or the uniqueness content-hash.
        """
        cell_line = [{'id': 'SIDM001', 'name': 'A', 'tissue': 'Breast', 'tissueId': 'X'}]
        df = pl.DataFrame([
            _raw_row(diseaseCellLines=cell_line),
            _raw_row(diseaseCellLines=cell_line, interactingTargetFromSourceId='OTHER'),
        ])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, [*UNIQUE_FIELDS, 'diseaseCellLines'])

        assert result.height == 2
        assert all(r['qualityControls'] == [] for r in result.to_dicts())

    def test_duplicate_rows_are_flagged_and_one_survives(self) -> None:
        df = pl.DataFrame([_raw_row(), _raw_row()])
        disease_lut, target_lut = _luts()

        result = _process_encore_evidence(df, disease_lut, target_lut, UNIQUE_FIELDS)

        assert sorted(result['qualityControls'].to_list()) == [[], ['Duplicated']]
