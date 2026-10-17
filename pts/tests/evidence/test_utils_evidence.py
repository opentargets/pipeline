"""Tests for the polars evidence post-processing.

Two things live here: the validation look-up-table builders (`luts.py`), and the
`Evidence` class -- harmonisation, entity validation, identifier assignment, uniqueness, dating,
scoring and direction of effect (`evidence.py`).
"""

from __future__ import annotations

import gzip
import hashlib
import json
from datetime import date
from pathlib import Path

import polars as pl
import polars_hash as plh
import pytest

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.utils.evidence import (
    DATE_COLUMNS,
    QC_COLUMN,
    VARIANT_HASH_LENGTH,
    Evidence,
    EvidenceFlags,
)
from pts.transformers.evidence.utils.hashing import UnsupportedIdentifierField, render_for_hash
from pts.transformers.evidence.utils.luts import (
    build_disease_lut,
    build_publication_lut,
    build_target_lut,
)

TARGET_SCHEMA = {
    'id': pl.String,
    'biotype': pl.String,
    'approvedSymbol': pl.String,
    'proteinIds': pl.List(pl.Struct({'id': pl.String, 'source': pl.String})),
    'hallmarks': pl.Struct({'attributes': pl.List(pl.Struct({'description': pl.String}))}),
}

EVIDENCE_SCHEMA = evidence_schema


def _write_parquet(directory: Path, frame: pl.DataFrame) -> str:
    """Write a frame as a single part file, the way the upstream steps lay a dataset out."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / '_SUCCESS').touch()
    frame.write_parquet(directory / 'part-00000.parquet')
    return str(directory)


def _write_ndjson(directory: Path, *parts: list[dict]) -> str:
    """Write one gzipped newline delimited json file per part."""
    directory.mkdir(parents=True, exist_ok=True)
    for index, rows in enumerate(parts):
        payload = '\n'.join(json.dumps(row) for row in rows).encode()
        (directory / f'part-{index:05d}.json.gz').write_bytes(gzip.compress(payload))
    return str(directory)


def _disease(directory: Path, rows: dict) -> str:
    return _write_parquet(
        directory, pl.DataFrame(rows, schema={'id': pl.String, 'obsoleteTerms': pl.List(pl.String)})
    )


def _target(directory: Path, rows: dict) -> str:
    return _write_parquet(directory, pl.DataFrame(rows, schema=TARGET_SCHEMA))


def _hallmarks(*descriptions: str | None) -> dict:
    return {'attributes': [{'description': d} for d in descriptions]}


# --------------------------------------------------------------------------- disease


def test_disease_lut_includes_obsolete_terms(tmp_path: Path) -> None:
    lut = build_disease_lut(_disease(tmp_path / 'disease', {'id': ['EFO_1'], 'obsoleteTerms': [['OLD_1', 'OLD_2']]}))

    assert set(lut['diseaseFromSourceMappedId']) == {'EFO_1', 'OLD_1', 'OLD_2'}
    assert set(lut['diseaseId']) == {'EFO_1'}


def test_disease_lut_handles_null_obsolete_terms(tmp_path: Path) -> None:
    lut = build_disease_lut(_disease(tmp_path / 'disease', {'id': ['EFO_1'], 'obsoleteTerms': [None]}))

    assert lut.to_dicts() == [{'diseaseId': 'EFO_1', 'diseaseFromSourceMappedId': 'EFO_1'}]


def test_disease_lut_keeps_duplicate_rows(tmp_path: Path) -> None:
    """Spark's `_prepare_disease_lut` has no `distinct()`, and 10 real 26.06 diseases hit this.

    Deduplicating would drop the second copy of the evidence the duplicate fans out,
    and that copy is what `validate_uniqueness` later flags as DUPLICATED. Dropping it
    here moves rows out of `failed_evidence`, so the duplicates have to survive.
    """
    path = _disease(tmp_path / 'disease', {'id': ['EFO_1'], 'obsoleteTerms': [['EFO_1', 'OLD_1']]})

    assert sorted(build_disease_lut(path)['diseaseFromSourceMappedId']) == ['EFO_1', 'EFO_1', 'OLD_1']


def test_disease_lut_order_does_not_depend_on_input_order(tmp_path: Path) -> None:
    rows = {'id': ['EFO_2', 'EFO_1', 'EFO_3'], 'obsoleteTerms': [['OLD'], ['OLD', 'OLD_1'], None]}
    reversed_rows = {k: v[::-1] for k, v in rows.items()}

    lut = build_disease_lut(_disease(tmp_path / 'a', rows))
    assert lut.equals(build_disease_lut(_disease(tmp_path / 'b', reversed_rows)))
    assert lut.filter(pl.col('diseaseFromSourceMappedId') == 'OLD')['diseaseId'].to_list() == ['EFO_1', 'EFO_2']


# --------------------------------------------------------------------------- target


def test_target_lut_keys_on_every_identifier(tmp_path: Path) -> None:
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': ['ABC1'],
            'proteinIds': [[{'id': 'P1', 'source': 'uniprot'}, {'id': 'P2', 'source': 'uniprot'}]],
            'hallmarks': [_hallmarks('oncogene role')],
        },
    )
    lut = build_target_lut(path)

    assert set(lut['targetFromSourceId']) == {'ENSG1', 'P1', 'P2', 'ABC1'}
    assert set(lut['TSorOncogene']) == {'oncogene'}
    assert set(lut['biotype']) == {'protein_coding'}


def test_target_lut_order_does_not_depend_on_input_order(tmp_path: Path) -> None:
    rows = {
        'id': ['ENSG2', 'ENSG1', 'ENSG3'],
        'biotype': ['protein_coding'] * 3,
        'approvedSymbol': ['SYM', 'SYM', 'OTHER'],
        'proteinIds': [[{'id': 'P1', 'source': 'uniprot'}], [{'id': 'P1', 'source': 'uniprot'}], None],
        'hallmarks': [None] * 3,
    }
    reversed_rows = {k: v[::-1] for k, v in rows.items()}

    lut = build_target_lut(_target(tmp_path / 'a', rows))
    assert lut.equals(build_target_lut(_target(tmp_path / 'b', reversed_rows)))
    assert lut.filter(pl.col('targetFromSourceId') == 'SYM')['targetId'].to_list() == ['ENSG1', 'ENSG2']


def test_target_lut_deduplicates_repeated_identifiers(tmp_path: Path) -> None:
    """Spark's `_prepare_target_lut` does `array_distinct` then `distinct`; both are needed."""
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': ['ENSG1'],
            'proteinIds': [[{'id': 'ENSG1', 'source': 'uniprot'}]],
            'hallmarks': [None],
        },
    )

    assert build_target_lut(path)['targetFromSourceId'].to_list() == ['ENSG1']


def test_target_lut_handles_missing_symbol_and_proteins(tmp_path: Path) -> None:
    """A null `proteinIds` nulls the whole `concat_list`, so it has to be filled first."""
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': [None],
            'proteinIds': [None],
            'hallmarks': [None],
        },
    )
    lut = build_target_lut(path)

    assert sorted(lut['targetFromSourceId'], key=lambda v: (v is None, v)) == ['ENSG1', None]
    assert set(lut['TSorOncogene']) == {None}


@pytest.mark.parametrize(
    ('hallmarks', 'expected'),
    [
        (_hallmarks('oncogene role'), 'oncogene'),
        (_hallmarks('tsg role'), 'tsg'),
        (_hallmarks('acts as oncogene and tsg'), 'bivalent'),
        (_hallmarks('oncogene role', 'tsg role'), 'bivalent'),
        (_hallmarks('ONCOGENE in caps'), 'oncogene'),
        (_hallmarks('unrelated annotation'), None),
        (_hallmarks(None), None),
        (_hallmarks(), None),
        (None, None),
    ],
)
def test_target_lut_cancer_gene_assessment(tmp_path: Path, hallmarks: dict | None, expected: str | None) -> None:
    path = _target(
        tmp_path / 'target',
        {
            'id': ['ENSG1'],
            'biotype': ['protein_coding'],
            'approvedSymbol': ['ABC1'],
            'proteinIds': [[]],
            'hallmarks': [hallmarks],
        },
    )

    assert build_target_lut(path)['TSorOncogene'].to_list() == [expected, expected]


# --------------------------------------------------------------------------- publication


def _publication(pmid: str | None, identifier: str, pmcid: str | None, date: str, source: str = 'MED') -> dict:
    return {'pmid': pmid, 'id': identifier, 'pmcid': pmcid, 'firstPublicationDate': date, 'source': source}


def test_publication_lut_keys_on_every_identifier(tmp_path: Path) -> None:
    path = _write_ndjson(tmp_path / 'literature', [_publication('123', 'MED_1', 'PMC1', '2020-01-01')])

    assert build_publication_lut(path).sort('publicationId').to_dicts() == [
        {'publicationDate': '2020-01-01', 'publicationId': '123'},
        {'publicationDate': '2020-01-01', 'publicationId': 'MED_1'},
        {'publicationDate': '2020-01-01', 'publicationId': 'PMC1'},
    ]


def test_publication_lut_drops_null_identifiers(tmp_path: Path) -> None:
    path = _write_ndjson(tmp_path / 'literature', [_publication(None, 'MED_1', None, '2020-01-01')])

    assert build_publication_lut(path)['publicationId'].to_list() == ['MED_1']


def test_publication_lut_filters_on_source(tmp_path: Path) -> None:
    path = _write_ndjson(
        tmp_path / 'literature',
        [
            _publication(None, 'MED_1', None, '2020-01-01', source='MED'),
            _publication(None, 'PPR_1', None, '2020-01-01', source='PPR'),
            _publication(None, 'AGR_1', None, '2020-01-01', source='AGR'),
            _publication(None, 'PAT_1', None, '2020-01-01', source='PAT'),
        ],
    )

    assert sorted(build_publication_lut(path)['publicationId']) == ['AGR_1', 'MED_1', 'PPR_1']


def test_publication_lut_deduplicates(tmp_path: Path) -> None:
    """`pmid` and `id` carry the same value for most MED records — 44% of the real rows."""
    path = _write_ndjson(tmp_path / 'literature', [_publication('123', '123', None, '2020-01-01')])

    assert build_publication_lut(path)['publicationId'].to_list() == ['123']


def test_publication_lut_reads_a_column_absent_from_the_leading_rows(tmp_path: Path) -> None:
    """Polars infers ndjson dtypes from a sample, and the real export has files whose first rows have no pmid.

    Inference then types the column Null, which fails the read outright, or, under
    `ignore_errors`, discards every value in it — 386,627 pmids in one file of the
    26.06 export. The reader pins the schema so the answer does not depend on where
    the nulls happen to fall.
    """
    rows = [_publication(None, f'MED_{i}', None, '2020-01-01') for i in range(200)]
    rows.append(_publication('999', 'MED_999', None, '2020-01-01'))
    path = _write_ndjson(tmp_path / 'literature', rows)

    assert '999' in set(build_publication_lut(path)['publicationId'])


def test_publication_lut_ignores_an_unpinned_column_typed_null_from_leading_rows(tmp_path: Path) -> None:
    """An UNPINNED column the lookup table never uses can still abort the whole read.

    Real 26.06 data: one export file has a `dateOfPublication` column (not one of `LITERATURE_
    SCHEMA`'s five pinned fields) that infers as Null from its leading rows -- absent there --
    then hits a non-null value ('2005 Oct') later. `schema_overrides=` only pins the columns it
    names; polars still infers every OTHER column in the file, so the read raises `ComputeError:
    got non-null value for NULL-typed column` on a column the table doesn't even select. `schema=`
    instead of `schema_overrides=` restricts parsing to exactly the five pinned columns, so an
    unpinned column's shape can never break the read -- this is the case that actually broke in
    production; `test_publication_lut_reads_a_column_absent_from_the_leading_rows` above only
    covers a PINNED column having this shape.
    """
    rows = [{**_publication(None, f'MED_{i}', None, '2020-01-01'), 'dateOfPublication': None} for i in range(200)]
    rows.append({**_publication('999', 'MED_999', None, '2020-01-01'), 'dateOfPublication': '2005 Oct'})
    path = _write_ndjson(tmp_path / 'literature', rows)

    assert '999' in set(build_publication_lut(path)['publicationId'])


def test_publication_lut_reads_every_part(tmp_path: Path) -> None:
    """Each part infers its own schema, so parts have to be read under one pinned schema."""
    path = _write_ndjson(
        tmp_path / 'literature',
        [_publication(None, 'MED_1', None, '2020-01-01')],
        [_publication('222', 'MED_2', None, '2021-01-01')],
    )

    assert sorted(build_publication_lut(path)['publicationId']) == ['222', 'MED_1', 'MED_2']


# --------------------------------------------------------------------------- reading


@pytest.mark.parametrize(
    ('builder', 'pattern'),
    [(build_disease_lut, '*.parquet'), (build_target_lut, '*.parquet'), (build_publication_lut, '*.json*')],
)
def test_builders_fail_loudly_on_an_empty_dataset(tmp_path: Path, builder, pattern: str) -> None:
    empty = tmp_path / 'empty'
    empty.mkdir()
    (empty / '_SUCCESS').touch()

    with pytest.raises(ValueError, match=f'no {pattern.replace("*", ".")} '):
        builder(str(empty))


# --------------------------------------------------------------------------- evidence


class TestRenderForHashStringAndList:
    def test_string_is_identity(self) -> None:
        df = pl.DataFrame({'v': ['a', '', None]})
        assert df.select(render_for_hash('v', pl.String).alias('o'))['o'].to_list() == ['a', '', None]

    def test_list_string_renders_like_spark(self) -> None:
        df = pl.DataFrame({'lit': [['1', '2'], [], None]}, schema={'lit': pl.List(pl.String)})
        got = df.select(render_for_hash('lit', pl.List(pl.String)).alias('v'))['v'].to_list()
        assert got == ['[1, 2]', '[]', None]

    def test_list_string_null_element_keeps_literal_null_token(self) -> None:
        # Measured against spark (precheck-parity.md Q1): a null element renders as the bare
        # token `null`, not as a dropped element -- `list.join` alone silently drops it.
        df = pl.DataFrame({'lit': [['a', None]]}, schema={'lit': pl.List(pl.String)})
        got = df.select(render_for_hash('lit', pl.List(pl.String)).alias('v'))['v'].to_list()
        assert got == ['[a, null]']


class TestRenderForHashFloat:
    """Float64 renders as java's `Double.toString` (spark parity in `tests/test_spark_strings.py`)."""

    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (1.0, '1.0'),
            (0.5, '0.5'),
            (-0.0, '-0.0'),
            (0.0, '0.0'),
            (100.0, '100.0'),
            (999999.9, '999999.9'),
            (0.001, '0.001'),
            (-5.5, '-5.5'),
        ],
    )
    def test_renders_the_common_range_exactly_as_spark_does(self, value: float, expected: str) -> None:
        df = pl.DataFrame({'v': [value]}, schema={'v': pl.Float64})
        assert df.select(render_for_hash('v', pl.Float64).alias('o'))['o'].to_list() == [expected]

    @pytest.mark.parametrize(
        ('value', 'spark_renders'),
        [
            (1e-10, '1.0E-10'),
            (0.0001, '1.0E-4'),
            (10000000.0, '1.0E7'),
            (123456789.123, '1.23456789123E8'),
            (1e15, '1.0E15'),
        ],
    )
    def test_renders_scientific_notation_as_spark_does(self, value: float, spark_renders: str) -> None:
        df = pl.DataFrame({'v': [value]}, schema={'v': pl.Float64})
        assert df.select(render_for_hash('v', pl.Float64).alias('o'))['o'].to_list() == [spark_renders]

    def test_float_hash_rendering_is_java(self) -> None:
        df = pl.DataFrame({'v': [1e-5]}, schema={'v': pl.Float64})
        assert df.select(render_for_hash('v', pl.Float64).alias('o'))['o'].to_list() == ['1.0E-5']

    def test_float32_is_widened_to_a_double(self) -> None:
        df = pl.DataFrame({'v': [0.1, 1e-5]}, schema={'v': pl.Float32})
        got = df.select(render_for_hash('v', pl.Float32).alias('o'))['o'].to_list()
        assert got == ['0.10000000149011612', '9.999999747378752E-6']

    def test_null_stays_null(self) -> None:
        df = pl.DataFrame({'v': [None]}, schema={'v': pl.Float64})
        assert df.select(render_for_hash('v', pl.Float64).alias('o'))['o'].to_list() == [None]

    @pytest.mark.parametrize(
        'value',
        [
            1.0,
            0.5,
            1e-10,
            123456789.123,
            1e15,
            1.7e16,
            1.5e20,
            1.7976931348623157e308,  # DBL_MAX
            5e-324,  # Double.MIN_VALUE, mantissa popcount 1
            8.095e-320,  # mantissa 16384 (2**14)
            1.66880539388046e-308,  # a real gene_burden.resourceScore, subnormal
            5.498e-320,  # a real gene_burden.resourceScore, subnormal
        ],
    )
    def test_every_finite_double_round_trips(self, value: float) -> None:
        # THE contract now. Includes the two real gene_burden subnormals and the extremes that
        # used to raise UnsupportedIdentifierField and block a whole datasource.
        rendered = pl.DataFrame({'v': [value]}, schema={'v': pl.Float64}).select(
            render_for_hash('v', pl.Float64).alias('o')
        )['o'][0]

        assert float(rendered) == value

    def test_distinct_values_keep_distinct_renderings(self) -> None:
        # What the de-duplication hash actually depends on: if two different doubles rendered
        # identically, they would rank as equal content and the survivor pick would be arbitrary.
        values = [1.0, 1.0000000000000002, 0.1, 0.30000000000000004, 5e-324, 1e-323, 1e15, 1e16]
        df = pl.DataFrame({'v': values}, schema={'v': pl.Float64})

        rendered = df.select(render_for_hash('v', pl.Float64).alias('o'))['o'].to_list()

        assert len(set(rendered)) == len(values)


class TestRenderForHashListOfStruct:
    CELL_TYPE = pl.List(pl.Struct({'id': pl.String, 'name': pl.String, 'tissue': pl.String, 'tissueId': pl.String}))

    def test_matches_measured_spark_rendering(self) -> None:
        # Measured against real spark (task-8-report.md): a struct element renders as
        # `{f1, f2, f3, f4}` (curly braces, no field names, comma-space separated), a null
        # field renders as the literal token `null`, and the whole thing nests inside the
        # usual `[...]` array rendering.
        df = pl.DataFrame(
            {
                'cells': [
                    [{'id': 'CL_1', 'name': 'HeLa', 'tissue': 'cervix', 'tissueId': 'UBERON_1'}],
                    [
                        {'id': 'CL_1', 'name': 'HeLa', 'tissue': 'cervix', 'tissueId': 'UBERON_1'},
                        {'id': 'CL_2', 'name': 'HEK293', 'tissue': None, 'tissueId': 'UBERON_2'},
                    ],
                    [],
                    None,
                    [{'id': None, 'name': None, 'tissue': None, 'tissueId': None}],
                ]
            },
            schema={'cells': self.CELL_TYPE},
        )
        got = df.select(render_for_hash('cells', self.CELL_TYPE).alias('v'))['v'].to_list()
        assert got == [
            '[{CL_1, HeLa, cervix, UBERON_1}]',
            '[{CL_1, HeLa, cervix, UBERON_1}, {CL_2, HEK293, null, UBERON_2}]',
            '[]',
            None,
            '[{null, null, null, null}]',
        ]

    def test_non_string_struct_field_raises(self) -> None:
        # Int64/Boolean struct fields are now measured and supported (task-9-report.md,
        # mutatedSamples/textMiningSentences/assays) -- Date stands in as a genuinely unmeasured
        # struct field dtype instead, to keep this pin meaningful.
        dtype = pl.List(pl.Struct({'id': pl.String, 'count': pl.Date}))
        df = pl.DataFrame({'v': [[{'id': 'a', 'count': date(2020, 1, 1)}]]}, schema={'v': dtype})
        with pytest.raises(UnsupportedIdentifierField, match='count'):
            df.select(render_for_hash('v', dtype).alias('o'))


class TestRenderForHashUnsupported:
    def test_unsupported_dtype_raises(self) -> None:
        # Boolean is now measured and supported (task-9-report.md) -- Date stands in as a
        # genuinely unmeasured top-level scalar dtype instead.
        df = pl.DataFrame({'v': [None]}, schema={'v': pl.Date})
        with pytest.raises(UnsupportedIdentifierField, match='Date'):
            df.select(render_for_hash('v', pl.Date).alias('o'))

    def test_unsupported_list_element_dtype_raises(self) -> None:
        # Int64 list elements are now measured and supported (task-9-report.md) -- Date stands in
        # as a genuinely unmeasured list-element dtype instead.
        df = pl.DataFrame({'v': [[None]]}, schema={'v': pl.List(pl.Date)})
        with pytest.raises(UnsupportedIdentifierField, match='Date'):
            df.select(render_for_hash('v', pl.List(pl.Date)).alias('o'))


class TestFlagNullQualityControls:
    """Correction 1: a null qualityControls column must normalise to [] in both branches."""

    def test_null_qc_condition_true_becomes_flag_list(self) -> None:
        lf = pl.LazyFrame({'diseaseFromSourceMappedId': ['NOPE'], 'datasourceId': ['x']}).with_columns(
            pl.lit(None, dtype=pl.List(pl.String)).alias(QC_COLUMN)
        )
        lut = pl.DataFrame(
            {'diseaseFromSourceMappedId': [], 'diseaseId': []},
            schema={'diseaseFromSourceMappedId': pl.String, 'diseaseId': pl.String},
        )
        out = Evidence(lf).validate_diseases(lut).lf.collect()
        assert out[QC_COLUMN].to_list() == [[EvidenceFlags.INVALID_DISEASE]]

    def test_null_qc_condition_false_becomes_empty_list_not_null(self) -> None:
        lf = pl.LazyFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'datasourceId': ['x']}).with_columns(
            pl.lit(None, dtype=pl.List(pl.String)).alias(QC_COLUMN)
        )
        lut = pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']})
        out = Evidence(lf).validate_diseases(lut).lf.collect()
        # Not [] AND not null-as-a-python-None: a null-qc row that fails neither the "has
        # flags" nor the "no flags" predicate would silently vanish from both valid and
        # failed evidence -- measured in precheck-parity.md Q2.
        assert out[QC_COLUMN].to_list() == [[]]

    def test_null_element_in_incoming_qc_sorts_last_like_spark(self) -> None:
        # polars `list.sort()` puts a null element first by default; spark's `array_sort` puts
        # it last. The flag text itself is never null (it comes from the EvidenceFlags enum), so
        # this exercises a null already present in an incoming qualityControls list.
        lf = pl.LazyFrame({'diseaseFromSourceMappedId': ['NOPE']}).with_columns(
            pl.Series(QC_COLUMN, [['z', None]], dtype=pl.List(pl.String))
        )
        lut = pl.DataFrame(
            {'diseaseFromSourceMappedId': [], 'diseaseId': []},
            schema={'diseaseFromSourceMappedId': pl.String, 'diseaseId': pl.String},
        )
        out = Evidence(lf).validate_diseases(lut).lf.collect()
        assert out[QC_COLUMN].to_list() == [[EvidenceFlags.INVALID_DISEASE, 'z', None]]


class TestValidateDiseases:
    def test_invalid_disease_is_flagged(self) -> None:
        lf = pl.LazyFrame({'diseaseFromSourceMappedId': ['EFO_1', 'NOPE'], 'datasourceId': ['x', 'x']})
        lut = pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']})
        out = Evidence(lf).validate_diseases(lut).lf.collect()
        assert out[QC_COLUMN].to_list() == [[], [EvidenceFlags.INVALID_DISEASE]]


    def test_fan_out_follows_lookup_order(self) -> None:
        lf = pl.LazyFrame({'diseaseFromSourceMappedId': ['OLD'] * 50 + ['EFO_9'] * 50, 'n': list(range(100))})
        for ids in (['EFO_1', 'EFO_2', 'EFO_3'], ['EFO_3', 'EFO_1', 'EFO_2']):
            lut = pl.DataFrame({'diseaseFromSourceMappedId': ['OLD'] * 3, 'diseaseId': ids})
            out = Evidence(lf).validate_diseases(lut).lf.collect()
            assert out['n'].to_list() == [n for n in range(50) for _ in ids] + list(range(50, 100))
            assert out['diseaseId'].to_list()[:150] == ids * 50


class TestValidateTarget:
    def test_invalid_target_is_flagged(self) -> None:
        lf = pl.LazyFrame({'targetFromSourceId': ['ENSG1', 'NOPE']})
        lut = pl.DataFrame(
            {'targetId': ['T1'], 'biotype': ['protein_coding'], 'targetFromSourceId': ['ENSG1']}
        )
        out = Evidence(lf).validate_target(lut).lf.collect()
        assert out[QC_COLUMN].to_list() == [[], [EvidenceFlags.INVALID_TARGET]]
        assert 'biotype' not in out.columns

    def test_fan_out_follows_lookup_order(self) -> None:
        lf = pl.LazyFrame({'targetFromSourceId': ['SYM'] * 50 + ['NOPE'] * 50, 'n': list(range(100))})
        for ids in (['T1', 'T2', 'T3'], ['T3', 'T1', 'T2']):
            lut = pl.DataFrame({'targetId': ids, 'biotype': ['protein_coding'] * 3, 'targetFromSourceId': ['SYM'] * 3})
            out = Evidence(lf).validate_target(lut).lf.collect()
            assert out['n'].to_list() == [n for n in range(50) for _ in ids] + list(range(50, 100))
            assert out['targetId'].to_list()[:150] == ids * 50

    def test_invalid_biotype_is_flagged(self) -> None:
        lf = pl.LazyFrame({'targetFromSourceId': ['ENSG1', 'ENSG2']})
        lut = pl.DataFrame(
            {
                'targetId': ['T1', 'T2'],
                'biotype': ['pseudogene', 'protein_coding'],
                'targetFromSourceId': ['ENSG1', 'ENSG2'],
            }
        )
        out = Evidence(lf).validate_target(lut, invalid_biotypes=['pseudogene']).lf.collect()
        assert out[QC_COLUMN].to_list() == [[EvidenceFlags.INVALID_BIOTYPE], []]

    # A prior version of this class had a
    # `test_no_invalid_biotypes_provided_skips_biotype_flag` test asserting `QC_COLUMN == [[]]`
    # with `invalid_biotypes` omitted. Deleted on review: `is_in([])` is always False, so a
    # version of `validate_target` that always applies the biotype `_flag` (rather than skipping
    # it when `invalid_biotypes` is falsy) produces the identical `[[]]` result -- there is no
    # assertion on the QC list itself that can distinguish "the branch was skipped" from "the
    # branch ran as a no-op", so the test could not fail against a broken implementation.
    # `test_invalid_target_is_flagged` already covers calling `validate_target` without
    # `invalid_biotypes` and dropping `biotype`.


class TestValidateDatasource:
    def test_filters_to_the_requested_datasource(self) -> None:
        lf = pl.LazyFrame({'datasourceId': ['a', 'b', 'a']})
        out = Evidence(lf).validate_datasource('a').lf.collect()
        assert out['datasourceId'].to_list() == ['a', 'a']


class TestAssignEvidenceIdentifier:
    def test_identifier_is_unique_per_distinct_field_combination(self) -> None:
        lf = pl.LazyFrame({
            'targetId': ['T1', 'T1', 'T2'],
            'datasourceId': ['d', 'd', 'd'],
        })
        out = Evidence(lf).assign_evidence_identifier(['targetId', 'datasourceId']).lf.collect()
        ids = out['id'].to_list()
        assert ids[0] == ids[1]
        assert ids[0] != ids[2]
        assert all(len(i) == 40 for i in ids)

    def test_field_boundaries_are_part_of_the_identifier(self) -> None:
        lf = pl.LazyFrame({'contrast': ['ab', 'a'], 'studyId': ['c', 'bc']})
        out = Evidence(lf).assign_evidence_identifier(['contrast', 'studyId']).lf.collect()
        assert out['id'][0] != out['id'][1]

    def test_missing_unique_field_is_silently_skipped(self) -> None:
        # Not every unique_fields entry from config.yaml is present on every datasource's
        # frame; assign_evidence_identifier must not raise, mirroring the pyspark
        # `if col in self.df.columns` guard.
        lf = pl.LazyFrame({'targetId': ['T1']})
        out = Evidence(lf).assign_evidence_identifier(['targetId', 'doesNotExist']).lf.collect()
        assert len(out['id'][0]) == 40

    def test_null_field_value_is_hashed_as_the_null_token(self) -> None:
        # coalesce(col, 'null') semantics: a genuinely-null unique field must not collide with
        # an empty string, and must be deterministic (the 'null' token, like spark's coalesce).
        lf = pl.LazyFrame({'targetId': [None, 'null']}, schema={'targetId': pl.String})
        out = Evidence(lf).assign_evidence_identifier(['targetId']).lf.collect()
        assert out['id'][0] == out['id'][1]

    def test_no_unique_field_present_hashes_the_empty_string_like_spark(self) -> None:
        # spark's concat_ws over no columns yields '', so the id is sha1('').
        lf = pl.LazyFrame({'targetId': ['T1']})
        out = Evidence(lf).assign_evidence_identifier(['doesNotExist']).lf.collect()
        assert out['id'][0] == 'da39a3ee5e6b4b0d3255bfef95601890afd80709'


class TestHarmonisation:
    def test_qc_column_added_when_missing(self) -> None:
        out = Evidence(pl.LazyFrame({'targetId': ['T1']})).lf.collect()
        assert out[QC_COLUMN].to_list() == [[]]
        assert out.schema[QC_COLUMN] == pl.List(pl.String)

    def test_qc_column_left_alone_when_already_present(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1']}).with_columns(pl.lit(['x']).alias(QC_COLUMN))
        out = Evidence(lf).lf.collect()
        assert out[QC_COLUMN].to_list() == [['x']]

    def test_column_not_in_target_schema_is_left_alone(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1'], 'notInSchema': [1]})
        out = Evidence(lf).lf.collect()
        assert out['notInSchema'].to_list() == [1]

    def test_type_mismatch_is_cast_to_the_schema_type(self) -> None:
        # pValueExponent is `long` in evidence.json; a frame producing it as a plain Int32 (or
        # any other numeric dtype) must be cast, not left as-is.
        lf = pl.LazyFrame({'pValueExponent': [1, 2]}, schema={'pValueExponent': pl.Int32})
        out = Evidence(lf).lf.collect()
        assert out.schema['pValueExponent'] == EVIDENCE_SCHEMA['pValueExponent'] == pl.Int64

    def test_struct_field_missing_from_source_is_added_as_null(self) -> None:
        # biomarkers.geneExpression is List(Struct({id, name})); dropping `name` from the
        # source must add it back as null, not raise or drop the whole struct.
        dtype = pl.Struct({'geneExpression': pl.List(pl.Struct({'id': pl.String}))})
        lf = pl.LazyFrame({'biomarkers': [{'geneExpression': [{'id': 'g1'}]}]}, schema={'biomarkers': dtype})
        out = Evidence(lf).lf.collect()
        assert out['biomarkers'][0]['geneExpression'][0] == {'id': 'g1', 'name': None}

    def test_source_only_struct_field_is_dropped(self) -> None:
        # urls is List(Struct({niceName, url})); `extra` has no home in the target schema and
        # must disappear, while `url` (present in both) survives and `niceName` (target-only)
        # is added as null.
        dtype = pl.Struct({'url': pl.String, 'extra': pl.Int64})
        lf = pl.LazyFrame({'urls': [[{'url': 'http://x', 'extra': 1}]]}, schema={'urls': pl.List(dtype)})
        out = Evidence(lf).lf.collect()
        assert out['urls'][0][0] == {'niceName': None, 'url': 'http://x'}

    def test_qualitycontrols_ends_as_list_string_even_when_source_prebuilds_it(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1']}).with_columns(
            pl.lit(['x']).cast(pl.List(pl.String)).alias(QC_COLUMN)
        )
        out = Evidence(lf).lf.collect()
        assert out.schema[QC_COLUMN] == pl.List(pl.String)


class TestSparkParityForCastToString:
    """Direct spark-vs-polars checks for `render_for_hash`.

    Runs both engines against the same data and diffs the output, rather than pinning a
    hand-copied expected string that could quietly drift from real spark.

    `Float64` leaves are checked against spark in `tests/test_spark_strings.py`.
    """

    def test_list_string_matches_spark(self, spark) -> None:
        values = [['1', '2'], [], None, ['a', None]]
        rows = spark.createDataFrame([(v,) for v in values], 'v ARRAY<STRING>').selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': pl.List(pl.String)})
        got = df.select(render_for_hash('v', pl.List(pl.String)).alias('v'))['v'].to_list()
        assert got == expected

    def test_list_struct_matches_spark(self, spark) -> None:
        dtype = pl.List(pl.Struct({'id': pl.String, 'name': pl.String, 'tissue': pl.String, 'tissueId': pl.String}))
        values = [
            [{'id': 'CL_1', 'name': 'HeLa', 'tissue': 'cervix', 'tissueId': 'UBERON_1'}],
            [
                {'id': 'CL_1', 'name': 'HeLa', 'tissue': 'cervix', 'tissueId': 'UBERON_1'},
                {'id': 'CL_2', 'name': 'HEK293', 'tissue': None, 'tissueId': 'UBERON_2'},
            ],
            [],
            None,
            [{'id': None, 'name': None, 'tissue': None, 'tissueId': None}],
        ]
        spark_schema = 'v ARRAY<STRUCT<id: STRING, name: STRING, tissue: STRING, tissueId: STRING>>'
        rows = spark.createDataFrame([(v,) for v in values], spark_schema).selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': dtype})
        got = df.select(render_for_hash('v', dtype).alias('v'))['v'].to_list()
        assert got == expected

    def test_int64_matches_spark(self, spark) -> None:
        values = [5, -5, 0, None]
        rows = spark.createDataFrame([(v,) for v in values], 'v LONG').selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': pl.Int64})
        got = df.select(render_for_hash('v', pl.Int64).alias('v'))['v'].to_list()
        assert got == expected

    def test_boolean_matches_spark(self, spark) -> None:
        values = [True, False, None]
        rows = spark.createDataFrame([(v,) for v in values], 'v BOOLEAN').selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': pl.Boolean})
        got = df.select(render_for_hash('v', pl.Boolean).alias('v'))['v'].to_list()
        assert got == expected

    def test_list_struct_with_int_boolean_leaves_matches_spark(self, spark) -> None:
        # The reviewer's exact examples (task-9-report.md): mutatedSamples/textMiningSentences
        # have Int64 struct leaves, assays has a Boolean one -- these used to abort
        # validate_uniqueness's plan build entirely (UnsupportedIdentifierField) before every
        # real evidence.json column was measured.
        dtype = pl.List(pl.Struct({'a': pl.String, 'b': pl.Int64, 'c': pl.Int64, 'd': pl.Int64}))
        values = [
            [{'a': 'SO_1', 'b': 3, 'c': 100, 'd': 7}],
            [{'a': None, 'b': None, 'c': 0, 'd': -5}],
        ]
        spark_schema = 'v ARRAY<STRUCT<a: STRING, b: LONG, c: LONG, d: LONG>>'
        rows = spark.createDataFrame([(v,) for v in values], spark_schema).selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': dtype})
        got = df.select(render_for_hash('v', dtype).alias('v'))['v'].to_list()
        assert got == expected

        dtype2 = pl.List(pl.Struct({'d': pl.String, 'e': pl.Boolean, 'f': pl.String}))
        values2 = [
            [{'d': 'd', 'e': True, 'f': 's'}],
            [{'d': None, 'e': False, 'f': None}],
        ]
        spark_schema2 = 'v ARRAY<STRUCT<d: STRING, e: BOOLEAN, f: STRING>>'
        rows2 = spark.createDataFrame([(v,) for v in values2], spark_schema2).selectExpr('CAST(v AS STRING) as s')
        expected2 = [r['s'] for r in rows2.collect()]
        df2 = pl.DataFrame({'v': values2}, schema={'v': dtype2})
        got2 = df2.select(render_for_hash('v', dtype2).alias('v'))['v'].to_list()
        assert got2 == expected2

    def test_bare_struct_matches_spark(self, spark) -> None:
        # biomarkers is a bare top-level Struct (not List(Struct)) -- the renderer that used to
        # be inlined inside the List branch could not handle this shape at all.
        dtype = pl.Struct({'x': pl.String, 'y': pl.String})
        values = [{'x': 'a', 'y': 'b'}, {'x': None, 'y': 'b'}, None]
        rows = spark.createDataFrame(
            [(v,) for v in values], 'v STRUCT<x: STRING, y: STRING>'
        ).selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': dtype})
        got = df.select(render_for_hash('v', dtype).alias('v'))['v'].to_list()
        assert got == expected

    def test_struct_of_list_struct_matches_spark(self, spark) -> None:
        # biomarkers's actual shape: a Struct whose fields are List(Struct) -- the reviewer's
        # structural requirement was that the struct renderer be extracted and made recursive
        # specifically because this two-level nesting exists in real evidence.json data. Includes
        # a null nested List(Struct) field, which spark renders as the bare token 'null', not
        # '[]' or an absent field.
        dtype = pl.Struct(
            {
                'geneExpression': pl.List(pl.Struct({'id': pl.String})),
                'geneticVariation': pl.List(pl.Struct({'id': pl.String})),
            }
        )
        values = [{'geneExpression': [{'id': 'g1'}], 'geneticVariation': None}]
        spark_schema = (
            'v STRUCT<geneExpression: ARRAY<STRUCT<id: STRING>>, geneticVariation: ARRAY<STRUCT<id: STRING>>>'
        )
        rows = spark.createDataFrame([(v,) for v in values], spark_schema).selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': dtype})
        got = df.select(render_for_hash('v', dtype).alias('v'))['v'].to_list()
        assert got == expected


class TestValidateUniqueness:
    def test_duplicates_all_but_one_are_flagged(self) -> None:
        lf = pl.LazyFrame({'id': ['a', 'a', 'b'], 'v': ['1', '2', '3']})
        out = Evidence(lf).validate_uniqueness().lf.collect().sort('v')
        flagged = [EvidenceFlags.DUPLICATED in qc for qc in out[QC_COLUMN]]
        assert sum(flagged) == 1

    def test_survivor_matches_an_independently_computed_content_hash(self) -> None:
        # the survivor has the lowest sha256, computed with hashlib, of the row's rendered content
        lf = pl.LazyFrame({'id': ['a', 'a'], 'v': ['1', '2']})
        ev = Evidence(lf)
        schema = ev.lf.collect_schema()
        parts = [render_for_hash(name, schema[name]).fill_null('null') for name in schema.names()]
        contents = ev.lf.select(pl.concat_str(parts, separator='\x01').alias('_content'), 'v').collect()
        digests = {row['v']: hashlib.sha256(row['_content'].encode()).hexdigest() for row in contents.to_dicts()}
        expected_survivor = min(digests, key=lambda v: digests[v])

        out = ev.validate_uniqueness().lf.collect()
        flagged = {row['v']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in out.to_dicts()}
        assert flagged[expected_survivor] is False
        assert all(flagged[v] for v in flagged if v != expected_survivor)

    def test_rerunning_picks_the_same_survivor(self) -> None:
        # The content hash has to be a pure function of the row's own content, not of row
        # position/partition order -- rerunning must always flag the same row. Kept alongside
        # test_survivor_matches_an_independently_computed_content_hash, which is the one that
        # actually pins WHICH row wins; this one just pins run-to-run stability.
        lf = pl.LazyFrame({'id': ['a', 'a', 'a'], 'v': ['1', '2', '3']})
        first = Evidence(lf).validate_uniqueness().lf.collect().sort('v')[QC_COLUMN].to_list()
        second = Evidence(lf).validate_uniqueness().lf.collect().sort('v')[QC_COLUMN].to_list()
        assert first == second

    def test_unique_rows_are_not_flagged(self) -> None:
        lf = pl.LazyFrame({'id': ['a', 'b', 'c'], 'v': ['1', '2', '3']})
        out = Evidence(lf).validate_uniqueness().lf.collect()
        assert all(EvidenceFlags.DUPLICATED not in qc for qc in out[QC_COLUMN])

    def test_int64_and_boolean_columns_no_longer_abort_the_plan(self) -> None:
        # Correction E: before extending render_for_hash, a bare Int64/Boolean column
        # (e.g. publicationYear, primaryProjectHit) raised UnsupportedIdentifierField at plan
        # build time -- validate_uniqueness could never run on a real evidence frame at all.
        lf = pl.LazyFrame(
            {'id': ['a', 'a'], 'n': [1, 2], 'flag': [True, False]},
            schema={'id': pl.String, 'n': pl.Int64, 'flag': pl.Boolean},
        )
        out = Evidence(lf).validate_uniqueness().lf.collect()
        assert sum(EvidenceFlags.DUPLICATED in qc for qc in out[QC_COLUMN]) == 1

    def test_id_column_itself_is_included_in_the_content_hash(self) -> None:
        # with id in the hash '3' survives; without it '1' would
        lf = pl.LazyFrame({'id': ['a', 'a'], 'v': ['1', '3']})
        out = Evidence(lf).validate_uniqueness().lf.collect()
        flagged = {row['v']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in out.to_dicts()}
        assert flagged == {'1': True, '3': False}

    def test_fields_are_joined_with_spark_separator(self) -> None:
        # ('ab', 'c') and ('a', 'bc') only hash differently with a separator between fields;
        # joined by U+0001, ('ab', 'c') has the lower digest and survives.
        lf = pl.LazyFrame({'id': ['x', 'x'], 'p': ['a', 'ab'], 'q': ['bc', 'c']})
        ev = Evidence(lf)
        schema = ev.lf.collect_schema()
        parts = [render_for_hash(name, schema[name]).fill_null('null') for name in schema.names()]
        contents = ev.lf.select(pl.concat_str(parts, separator='\x01')).collect().to_series().to_list()
        assert contents == ['x\x01a\x01bc\x01[]', 'x\x01ab\x01c\x01[]']
        digests = [hashlib.sha256(c.encode()).hexdigest() for c in contents]
        assert digests[1] < digests[0]

        out = ev.validate_uniqueness().lf.collect()
        flagged = {row['p']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in out.to_dicts()}
        assert flagged == {'a': True, 'ab': False}

    def test_missing_id_raises_clearly(self) -> None:
        # Fix round 1, minor: pyspark's @required_columns(['id']) was never ported; without an
        # explicit check a missing id leaks a raw ColumnNotFoundError from deep inside
        # .over('id', ...) instead of naming the actual problem.
        lf = pl.LazyFrame({'v': ['1']})
        with pytest.raises(ValueError, match="'id'"):
            Evidence(lf).validate_uniqueness()

    def test_content_hash_matches_spark_sha2_256(self, spark) -> None:
        # Confirms the piece the survivor-parity argument in the docstring depends on: spark's
        # sha2(col, 256) and polars-hash's chash.sha2_256() produce the identical hex digest for
        # the identical input string -- not just "both call themselves SHA-256".
        value = 'aid_and_some_content_1'
        row = spark.createDataFrame([(value,)], 'v STRING').selectExpr('sha2(v, 256) as h').collect()[0]
        got = pl.DataFrame({'v': [value]}).select(plh.col('v').chash.sha2_256().alias('h'))['h'][0]
        assert got == row['h']


class TestResolvePublicationDate:
    def test_earliest_publication_date_wins(self) -> None:
        lf = pl.LazyFrame({'id': ['e1'], 'literature': [['P2', 'P1']]})
        lut = pl.DataFrame({'publicationId': ['P1', 'P2'], 'publicationDate': ['2001-01-01', '1999-01-01']})
        out = Evidence(lf).resolve_publication_date(lut).lf.collect()
        assert out['publicationDate'].to_list() == ['1999-01-01']

    def test_missing_literature_column_is_a_no_op(self) -> None:
        lf = pl.LazyFrame({'id': ['e1']})
        lut = pl.DataFrame(
            {'publicationId': [], 'publicationDate': []},
            schema={'publicationId': pl.String, 'publicationDate': pl.String},
        )
        out = Evidence(lf).resolve_publication_date(lut).lf.collect()
        assert 'publicationDate' not in out.columns

    def test_publication_id_is_uppercased_and_trimmed_before_lookup(self) -> None:
        lf = pl.LazyFrame({'id': ['e1'], 'literature': [[' p1 ']]})
        lut = pl.DataFrame({'publicationId': ['P1'], 'publicationDate': ['2020-01-01']})
        out = Evidence(lf).resolve_publication_date(lut).lf.collect()
        assert out['publicationDate'].to_list() == ['2020-01-01']

    def test_trim_only_strips_ascii_space_like_spark(self) -> None:
        # Correction C: spark's `trim` strips the ASCII space only; the no-argument polars
        # `str.strip_chars()` also strips tab/newline/non-breaking-space -- which would collapse
        # two distinct publication ids onto one lookup key. A tab-padded id must NOT match a
        # lookup keyed on the untouched (tab-free) id.
        lf = pl.LazyFrame({'id': ['e1'], 'literature': [['\tP1\t']]})
        lut = pl.DataFrame({'publicationId': ['P1'], 'publicationDate': ['2020-01-01']})
        out = Evidence(lf).resolve_publication_date(lut).lf.collect()
        assert out['publicationDate'].to_list() == [None]

    def test_no_matching_publication_leaves_date_null(self) -> None:
        lf = pl.LazyFrame({'id': ['e1'], 'literature': [['NOPE']]})
        lut = pl.DataFrame({'publicationId': ['P1'], 'publicationDate': ['2020-01-01']})
        out = Evidence(lf).resolve_publication_date(lut).lf.collect()
        assert out['publicationDate'].to_list() == [None]

    def test_missing_id_raises_clearly(self) -> None:
        # Fix round 1, minor: pyspark's @required_columns(['id']) was never ported here either.
        lf = pl.LazyFrame({'literature': [['P1']]})
        lut = pl.DataFrame({'publicationId': ['P1'], 'publicationDate': ['2020-01-01']})
        with pytest.raises(ValueError, match="'id'"):
            Evidence(lf).resolve_publication_date(lut)


class TestResolveEvidenceDate:
    def test_earliest_of_the_present_date_columns_wins(self) -> None:
        lf = pl.LazyFrame({'publicationDate': ['2001-01-01'], 'curationDate': ['1999-01-01']})
        out = Evidence(lf).resolve_evidence_date().lf.collect()
        assert out['evidenceDate'].to_list() == ['1999-01-01']

    def test_a_null_date_column_does_not_null_the_whole_min(self) -> None:
        # Unlike spark's array_min (null-propagating, so the pyspark implementation pre-filters),
        # pl.min_horizontal already ignores nulls -- a null studyStartDate must not blank out an
        # otherwise-present publicationDate.
        lf = pl.LazyFrame(
            {'publicationDate': ['2001-01-01'], 'studyStartDate': [None]},
            schema={'publicationDate': pl.String, 'studyStartDate': pl.String},
        )
        out = Evidence(lf).resolve_evidence_date().lf.collect()
        assert out['evidenceDate'].to_list() == ['2001-01-01']

    def test_no_date_columns_present_still_adds_a_null_column(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1']})
        out = Evidence(lf).resolve_evidence_date().lf.collect()
        assert out['evidenceDate'].to_list() == [None]
        assert out.schema['evidenceDate'] == pl.String

    def test_covers_all_date_columns(self) -> None:
        assert DATE_COLUMNS == ['publicationDate', 'curationDate', 'studyStartDate', 'releaseDate', 'evidenceDate']

    def test_existing_evidence_date_takes_part(self) -> None:
        lf = pl.LazyFrame({
            'publicationDate': ['2020-01-01', None, None],
            'curationDate': ['2021-01-01', '2019-06-01', None],
            'evidenceDate': ['2019-01-01', '2022-01-01', '2018-01-01'],
        })
        out = Evidence(lf).resolve_evidence_date().lf.collect()
        assert out['evidenceDate'].to_list() == ['2019-01-01', '2019-06-01', '2018-01-01']


class TestCalculateEvidenceScore:
    def test_score_out_of_range_is_flagged(self) -> None:
        lf = pl.LazyFrame({'resourceScore': [50.0, 500.0]})
        out = Evidence(lf).calculate_evidence_score(pl.col('resourceScore') / 100.0).lf.collect()
        assert out['score'].to_list() == [0.5, 5.0]
        assert [EvidenceFlags.NO_VALID_SCORE in qc for qc in out[QC_COLUMN]] == [False, True]

    def test_none_expression_raises_rather_than_silently_skipping(self) -> None:
        # unlike the direction methods, None is not optional for a score
        lf = pl.LazyFrame({'resourceScore': [0.5]})
        with pytest.raises(ValueError, match='score_expression'):
            Evidence(lf).calculate_evidence_score(None)

    def test_negative_score_is_flagged(self) -> None:
        lf = pl.LazyFrame({'resourceScore': [-0.1]})
        out = Evidence(lf).calculate_evidence_score(pl.col('resourceScore')).lf.collect()
        assert out[QC_COLUMN].to_list() == [[EvidenceFlags.NO_VALID_SCORE]]

    def test_missing_score_is_flagged(self) -> None:
        lf = pl.LazyFrame({'resourceScore': [None]}, schema={'resourceScore': pl.Float64})
        out = Evidence(lf).calculate_evidence_score(pl.col('resourceScore')).lf.collect()
        assert out[QC_COLUMN].to_list() == [[EvidenceFlags.NO_VALID_SCORE]]

    def test_unconvertible_value_casts_to_null_rather_than_raising(self) -> None:
        # Correction D: spark's cast yields null on an unconvertible value; a strict polars cast
        # raises instead -- strict=False is required to mirror spark rather than crash the step.
        lf = pl.LazyFrame({'resourceScore': ['not-a-number']})
        out = Evidence(lf).calculate_evidence_score(pl.col('resourceScore').cast(pl.String)).lf.collect()
        assert out['score'].to_list() == [None]
        assert out[QC_COLUMN].to_list() == [[EvidenceFlags.NO_VALID_SCORE]]


class TestAssignDirectionOnTrait:
    def test_expression_is_applied(self) -> None:
        lf = pl.LazyFrame({'diseaseId': ['EFO_1', None]})
        expr = pl.when(pl.col('diseaseId').is_not_null()).then(pl.lit('risk')).otherwise(None)
        out = Evidence(lf).assign_direction_on_trait(expr).lf.collect()
        assert out['directionOnTrait'].to_list() == ['risk', None]

    def test_none_expression_is_a_no_op(self) -> None:
        lf = pl.LazyFrame({'diseaseId': ['EFO_1']})
        out = Evidence(lf).assign_direction_on_trait(None).lf.collect()
        assert 'directionOnTrait' not in out.columns


class TestAssignDirectionOnTarget:
    def test_expression_is_applied(self) -> None:
        lf = pl.LazyFrame({'diseaseId': ['EFO_1', None]})
        expr = pl.when(pl.col('diseaseId').is_not_null()).then(pl.lit('LoF')).otherwise(None)
        out = Evidence(lf).assign_direction_on_target(expr, None).lf.collect()
        assert out['directionOnTarget'].to_list() == ['LoF', None]

    def test_none_expression_is_a_no_op(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1']})
        out = Evidence(lf).assign_direction_on_target(None, None).lf.collect()
        assert 'directionOnTarget' not in out.columns

    def test_target_lut_is_joined_and_dropped_afterwards(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1', 'T2']})
        lut = pl.DataFrame({'targetId': ['T1'], 'TSorOncogene': ['oncogene']})
        expr = pl.col('TSorOncogene').replace_strict({'oncogene': 'GoF', 'tsg': 'LoF'}, default=None)
        out = Evidence(lf).assign_direction_on_target(expr, lut).lf.collect()
        assert out['directionOnTarget'].to_list() == ['GoF', None]
        assert 'TSorOncogene' not in out.columns
        assert 'actionType' not in out.columns


class TestHashLongVariantIdentifiers:
    """Correction B: the null branch is reachable only when variantId itself is null.

    Every row is one of the team-lead's measured spark values (task-9-report.md).
    """

    def test_short_matching_id_is_unchanged(self) -> None:
        lf = pl.LazyFrame({'variantId': ['1_12345_A_T']})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        assert out['variantId'].to_list() == ['1_12345_A_T']

    def test_long_matching_id_is_hashed_with_chr_and_pos(self) -> None:
        variant_id = '1_12345_' + 'A' * 200 + '_' + 'T' * 200
        assert len(variant_id) == 409
        lf = pl.LazyFrame({'variantId': [variant_id]})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        got = out['variantId'][0]
        assert got.startswith('OTVAR_1_12345_')
        assert got != variant_id

    def test_short_non_matching_id_is_unchanged(self) -> None:
        lf = pl.LazyFrame({'variantId': ['rs12345']})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        assert out['variantId'].to_list() == ['rs12345']

    def test_long_non_matching_id_gets_triple_underscore(self) -> None:
        # concat_ws('_', 'OTVAR', '', '', md5) joins two empty strings -- the extraction is ''
        # (not null) for a non-match on a non-null input, per correction B.
        variant_id = 'rs' + '9' * 400
        assert len(variant_id) == 402
        lf = pl.LazyFrame({'variantId': [variant_id]})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        got = out['variantId'][0]
        assert got.startswith('OTVAR___')
        assert not got.startswith('OTVAR____')

    def test_empty_string_id_is_unchanged(self) -> None:
        lf = pl.LazyFrame({'variantId': ['']})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        assert out['variantId'].to_list() == ['']

    def test_null_id_stays_null(self) -> None:
        lf = pl.LazyFrame({'variantId': [None]}, schema={'variantId': pl.String})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        assert out['variantId'].to_list() == [None]

    def test_missing_variantid_column_is_a_no_op(self) -> None:
        lf = pl.LazyFrame({'targetId': ['T1']})
        out = Evidence(lf).hash_long_variant_identifiers().lf.collect()
        assert 'variantId' not in out.columns

    def test_hash_length_constant_matches_pyspark(self) -> None:
        assert VARIANT_HASH_LENGTH == 300

    def test_md5_digest_matches_spark(self, spark) -> None:
        # Fix round 1, minor: previous coverage only pinned the 'OTVAR_' prefix, not the digest
        # content itself -- confirms polars-hash's nchash.md5() and spark's md5() agree byte for
        # byte on the same input, not just that both call themselves md5.
        variant_id = '1_12345_' + 'A' * 200 + '_' + 'T' * 200
        row = spark.createDataFrame([(variant_id,)], 'v STRING').selectExpr('md5(v) as h').collect()[0]
        got = pl.DataFrame({'v': [variant_id]}).select(plh.col('v').nchash.md5().alias('h'))['h'][0]
        assert got == row['h']


class TestValidAndInvalid:
    def test_valid_and_invalid_split(self) -> None:
        lf = pl.LazyFrame({QC_COLUMN: [[], ['x']]}, schema={QC_COLUMN: pl.List(pl.String)})
        ev = Evidence(lf)
        assert ev.valid().collect().height == 1
        assert ev.invalid().collect().height == 1

    def test_valid_and_invalid_partition_every_row_exactly_once(self) -> None:
        # An 'id' per row, checked by value rather than by count: a `valid()` that returned every
        # row and an `invalid()` that returned none would also sum to 4 without actually
        # partitioning anything (test_valid_and_invalid_split already catches that specific
        # implementation via separate per-side counts, but this test's own name promised more).
        lf = pl.LazyFrame(
            {'id': ['r0', 'r1', 'r2', 'r3'], QC_COLUMN: [[], ['x'], [], ['y', 'z']]},
            schema={'id': pl.String, QC_COLUMN: pl.List(pl.String)},
        )
        ev = Evidence(lf)
        valid_ids = set(ev.valid().collect()['id'].to_list())
        invalid_ids = set(ev.invalid().collect()['id'].to_list())
        assert valid_ids == {'r0', 'r2'}
        assert invalid_ids == {'r1', 'r3'}
        assert valid_ids.isdisjoint(invalid_ids)


def test_uniqueness_survivor_does_not_depend_on_row_order() -> None:
    rows = pl.DataFrame({
        'datasourceId': ['ds'] * 6,
        'targetFromSourceId': ['T1'] * 6,
        'diseaseFromSourceMappedId': ['D1'] * 6,
        'resourceScore': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6],
    })

    def survivors(frame: pl.DataFrame) -> list[float]:
        ev = Evidence(frame.lazy()).assign_evidence_identifier(['targetFromSourceId']).validate_uniqueness()
        return sorted(ev.valid().collect()['resourceScore'].to_list())

    assert survivors(rows) == survivors(rows.reverse()) == survivors(rows.sample(fraction=1.0, shuffle=True, seed=1))
