"""Tests for the polars otar transformer."""

from __future__ import annotations

import polars as pl
import pytest

from pts.transformers.otar import _generate_otar_info, _spark_string_to_boolean


def _disease(*rows: tuple[str, list[str]]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema={'id': pl.String, 'ancestors': pl.List(pl.String)}, orient='row')


def _meta(*rows: tuple[str, str, str, str]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=['otar_code', 'project_name', 'project_status', 'integrates_in_PPP'],
        orient='row',
    )


def _lookup(*rows: tuple[str, str | None]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=['otar_code', 'efo_disease_id'], orient='row')


class TestSparkStringToBoolean:
    """Spark's cast(string as boolean) is permissive; polars refuses it outright."""

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            ('Y', True),
            ('y', True),
            ('T', True),
            ('true', True),
            ('TRUE', True),
            ('yes', True),
            ('1', True),
            (' Y ', True),
            ('N', False),
            ('n', False),
            ('F', False),
            ('false', False),
            ('no', False),
            ('0', False),
            ('maybe', None),
            ('', None),
            (None, None),
        ],
    )
    def test_matches_spark(self, value: str | None, expected: bool | None) -> None:
        df = pl.DataFrame({'v': [value]}, schema={'v': pl.String})
        assert df.select(_spark_string_to_boolean(pl.col('v')).alias('b'))['b'][0] is expected


class TestGenerateOtarInfo:
    def test_propagates_to_ancestors(self) -> None:
        result = _generate_otar_info(
            _disease(('EFO:0001', ['EFO:ROOT'])),
            _meta(('OTAR001', 'Project Alpha', 'Active', 'N')),
            _lookup(('OTAR001', 'EFO:0001')),
        )
        assert set(result['efo_id']) == {'EFO:0001', 'EFO:ROOT'}

    def test_includes_reference_url(self) -> None:
        result = _generate_otar_info(
            _disease(('EFO:0001', [])),
            _meta(('OTAR001', 'Project Alpha', 'Active', 'N')),
            _lookup(('OTAR001', 'EFO:0001')),
        )
        project = result['projects'][0][0]
        assert project['reference'] == 'http://home.opentargets.org/OTAR001'

    def test_y_and_n_become_booleans(self) -> None:
        result = _generate_otar_info(
            _disease(('EFO:0001', []), ('EFO:0002', [])),
            _meta(('OTAR001', 'Alpha', 'Active', 'Y'), ('OTAR002', 'Beta', 'Closed', 'N')),
            _lookup(('OTAR001', 'EFO:0001'), ('OTAR002', 'EFO:0002')),
        )
        flags = {r['efo_id']: r['projects'][0]['integrates_data_PPP'] for r in result.iter_rows(named=True)}
        assert flags == {'EFO:0001': True, 'EFO:0002': False}

    def test_unmapped_project_is_dropped(self) -> None:
        """A project with a null efo_disease_id contributes nothing."""
        result = _generate_otar_info(
            _disease(('EFO:0001', [])),
            _meta(('OTAR001', 'Alpha', 'Active', 'Y'), ('OTAR002', 'Beta', 'Active', 'Y')),
            _lookup(('OTAR001', 'EFO:0001'), ('OTAR002', None)),
        )
        codes = {p['otar_code'] for row in result['projects'] for p in row}
        assert codes == {'OTAR001'}

    def test_mapping_to_unknown_disease_is_dropped(self) -> None:
        result = _generate_otar_info(
            _disease(('EFO:0001', [])),
            _meta(('OTAR001', 'Alpha', 'Active', 'Y')),
            _lookup(('OTAR001', 'EFO:NOT_IN_INDEX')),
        )
        assert result.height == 0

    def test_projects_are_sorted_by_otar_code(self) -> None:
        result = _generate_otar_info(
            _disease(('EFO:0001', []), ('EFO:0002', []), ('EFO:0003', [])),
            _meta(
                ('OTAR003', 'Gamma', 'Active', 'Y'),
                ('OTAR001', 'Alpha', 'Active', 'Y'),
                ('OTAR002', 'Beta', 'Active', 'Y'),
            ),
            _lookup(('OTAR003', 'EFO:0001'), ('OTAR001', 'EFO:0001'), ('OTAR002', 'EFO:0001')),
        )
        row = result.row(by_predicate=pl.col('efo_id') == 'EFO:0001', named=True)
        assert [p['otar_code'] for p in row['projects']] == ['OTAR001', 'OTAR002', 'OTAR003']

    def test_shared_ancestor_collects_every_project(self) -> None:
        result = _generate_otar_info(
            _disease(('EFO:0001', ['EFO:ROOT']), ('EFO:0002', ['EFO:ROOT'])),
            _meta(('OTAR001', 'Alpha', 'Active', 'Y'), ('OTAR002', 'Beta', 'Closed', 'N')),
            _lookup(('OTAR001', 'EFO:0001'), ('OTAR002', 'EFO:0002')),
        )
        row = result.row(by_predicate=pl.col('efo_id') == 'EFO:ROOT', named=True)
        assert [p['otar_code'] for p in row['projects']] == ['OTAR001', 'OTAR002']

    def test_duplicate_projects_are_deduplicated(self) -> None:
        """Two mappings from one project onto the same ancestor yield one entry."""
        result = _generate_otar_info(
            _disease(('EFO:0001', ['EFO:ROOT']), ('EFO:0002', ['EFO:ROOT'])),
            _meta(('OTAR001', 'Alpha', 'Active', 'Y')),
            _lookup(('OTAR001', 'EFO:0001'), ('OTAR001', 'EFO:0002')),
        )
        row = result.row(by_predicate=pl.col('efo_id') == 'EFO:ROOT', named=True)
        assert [p['otar_code'] for p in row['projects']] == ['OTAR001']

    def test_output_is_stable_across_repeated_builds(self) -> None:
        disease = _disease(*[(f'EFO:{i:04d}', ['EFO:ROOT']) for i in range(20)])
        meta = _meta(*[(f'OTAR{i:03d}', f'P{i}', 'Active', 'Y') for i in range(20)])
        lookup = _lookup(*[(f'OTAR{i:03d}', f'EFO:{i:04d}') for i in range(20)])
        assert _generate_otar_info(disease, meta, lookup).equals(_generate_otar_info(disease, meta, lookup))
