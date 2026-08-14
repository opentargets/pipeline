"""Tests for the polars evidence post-processing.

Two things live here: the validation look-up-table builders (`validation_lut.py`), and the
`Evidence` class -- harmonisation, entity validation and identifier assignment (`evidence.py`).
"""

from __future__ import annotations

import gzip
import json
import random
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.utils.evidence import (
    QC_COLUMN,
    Evidence,
    EvidenceFlags,
    UnsupportedIdentifierField,
    spark_cast_to_string,
)
from pts.transformers.utils.schemas import load_spark_schema_as_polars
from pts.transformers.utils.validation_lut import (
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

EVIDENCE_SCHEMA = load_spark_schema_as_polars('evidence.json')


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


class TestSparkCastToStringStringAndList:
    def test_string_is_identity(self) -> None:
        df = pl.DataFrame({'v': ['a', '', None]})
        assert df.select(spark_cast_to_string('v', pl.String).alias('o'))['o'].to_list() == ['a', '', None]

    def test_list_string_renders_like_spark(self) -> None:
        df = pl.DataFrame({'lit': [['1', '2'], [], None]}, schema={'lit': pl.List(pl.String)})
        got = df.select(spark_cast_to_string('lit', pl.List(pl.String)).alias('v'))['v'].to_list()
        assert got == ['[1, 2]', '[]', None]

    def test_list_string_null_element_keeps_literal_null_token(self) -> None:
        # Measured against spark (precheck-parity.md Q1): a null element renders as the bare
        # token `null`, not as a dropped element -- `list.join` alone silently drops it.
        df = pl.DataFrame({'lit': [['a', None]]}, schema={'lit': pl.List(pl.String)})
        got = df.select(spark_cast_to_string('lit', pl.List(pl.String)).alias('v'))['v'].to_list()
        assert got == ['[a, null]']


class TestSparkCastToStringFloat:
    # Every value here (and its expected rendering) was measured against real spark; see
    # task-8-report.md. Spark's `CAST(DOUBLE AS STRING)` matches Java's `Double.toString`:
    # plain decimal for 0.001 <= |x| < 1e7, scientific `d.dddEn` outside that range.
    @pytest.mark.parametrize(
        ('value', 'expected'),
        [
            (1.0, '1.0'),
            (0.5, '0.5'),
            (1e-10, '1.0E-10'),
            (123456789.123, '1.23456789123E8'),
            (-0.0, '-0.0'),
            (0.0, '0.0'),
            (100.0, '100.0'),
            (999999.9, '999999.9'),
            (10000000.0, '1.0E7'),
            (0.001, '0.001'),
            (0.0001, '1.0E-4'),
            (-5.5, '-5.5'),
        ],
    )
    def test_matches_measured_spark_rendering(self, value: float, expected: str) -> None:
        df = pl.DataFrame({'v': [value]}, schema={'v': pl.Float64})
        assert df.select(spark_cast_to_string('v', pl.Float64).alias('o'))['o'].to_list() == [expected]

    def test_null_stays_null(self) -> None:
        df = pl.DataFrame({'v': [None]}, schema={'v': pl.Float64})
        assert df.select(spark_cast_to_string('v', pl.Float64).alias('o'))['o'].to_list() == [None]

    def test_boundary_values_still_render_exactly(self) -> None:
        # Regression pin for the reviewer's finding against an earlier, over-conservative
        # `1e15` cutoff: these values render exactly and must not be pushed into a raise (or a
        # non-exact fallback) by too wide a guard.
        values = [1e15, 1.5e16, 1.6999e16]
        expected = ['1.0E15', '1.5E16', '1.6999E16']
        df = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
        assert df.select(spark_cast_to_string('v', pl.Float64).alias('o'))['o'].to_list() == expected

    def test_large_magnitude_raises_rather_than_guess(self) -> None:
        # Measured (task-8-report.md): a 200,000-sample dense fuzz run against real spark in
        # [1e15, 2e16) found the java rendering algorithm's first non-shortest-digit mismatch at
        # 1.801526131658083e16 (spark '1.8029536657783832E16' vs shortest round-trip
        # '1.802953665778383E16' for a nearby value) -- and the onset is not a clean cutoff, only
        # some values past it mismatch. `1.7e16` is a cutoff with measured headroom (0 mismatches
        # in 85,297 + 150,000 dense samples below it), not the literal boundary. No real
        # resourceScore is ever this large, so raising here costs nothing in practice, and a
        # wrong-but-plausible id is worse than an unrunnable datasource that never actually needs
        # this path.
        df = pl.DataFrame({'v': [1.7e16, 1.5e20, 1.7976931348623157e308]}, schema={'v': pl.Float64})
        with pytest.raises(UnsupportedIdentifierField, match=r'1\.7e16'):
            df.select(spark_cast_to_string('v', pl.Float64).alias('o'))

    def test_subnormal_raises_rather_than_guess(self) -> None:
        # Measured (task-8-report.md): java's legacy Double.toString algorithm is not truly
        # shortest-round-trip for subnormals (e.g. Double.MIN_VALUE renders '4.9E-324' in java,
        # but python's shortest round-trip repr is '5e-324'). No evidence.json double field ever
        # reaches this range, so refuse loudly instead of silently emitting a value that could
        # diverge from spark and change an evidence id.
        df = pl.DataFrame({'v': [5e-324]}, schema={'v': pl.Float64})
        with pytest.raises(UnsupportedIdentifierField, match='subnormal'):
            df.select(spark_cast_to_string('v', pl.Float64).alias('o'))


class TestSparkCastToStringListOfStruct:
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
        got = df.select(spark_cast_to_string('cells', self.CELL_TYPE).alias('v'))['v'].to_list()
        assert got == [
            '[{CL_1, HeLa, cervix, UBERON_1}]',
            '[{CL_1, HeLa, cervix, UBERON_1}, {CL_2, HEK293, null, UBERON_2}]',
            '[]',
            None,
            '[{null, null, null, null}]',
        ]

    def test_non_string_struct_field_raises(self) -> None:
        # No evidence.json List(Struct) unique_fields type has a non-String field
        # (diseaseCellLines, the only one, is all-String) -- unmeasured against spark, so refuse
        # rather than guess at a rendering.
        dtype = pl.List(pl.Struct({'id': pl.String, 'count': pl.Int64}))
        df = pl.DataFrame({'v': [[{'id': 'a', 'count': 1}]]}, schema={'v': dtype})
        with pytest.raises(UnsupportedIdentifierField, match='count'):
            df.select(spark_cast_to_string('v', dtype).alias('o'))


class TestSparkCastToStringUnsupported:
    def test_unsupported_dtype_raises(self) -> None:
        df = pl.DataFrame({'v': [True]})
        with pytest.raises(UnsupportedIdentifierField, match='Boolean'):
            df.select(spark_cast_to_string('v', pl.Boolean).alias('o'))

    def test_unsupported_list_element_dtype_raises(self) -> None:
        df = pl.DataFrame({'v': [[1, 2]]}, schema={'v': pl.List(pl.Int64)})
        with pytest.raises(UnsupportedIdentifierField, match='Int64'):
            df.select(spark_cast_to_string('v', pl.List(pl.Int64)).alias('o'))


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


class TestValidateTarget:
    def test_invalid_target_is_flagged(self) -> None:
        lf = pl.LazyFrame({'targetFromSourceId': ['ENSG1', 'NOPE']})
        lut = pl.DataFrame(
            {'targetId': ['T1'], 'biotype': ['protein_coding'], 'targetFromSourceId': ['ENSG1']}
        )
        out = Evidence(lf).validate_target(lut).lf.collect()
        assert out[QC_COLUMN].to_list() == [[], [EvidenceFlags.INVALID_TARGET]]
        assert 'biotype' not in out.columns

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
        # pl.concat_str([]) raises ComputeError on an empty expression list; spark's
        # concat_ws('') over zero columns is well-defined and yields '', so the id must be
        # sha1(''), not a crash. Unreachable with today's config.yaml, but a raw ComputeError
        # leaking out of assign_evidence_identifier would be a real regression if that changes.
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
    """Direct spark-vs-polars checks for `spark_cast_to_string`.

    Runs both engines against the same data and diffs the output, rather than pinning a
    hand-copied expected string that could quietly drift from real spark.
    """

    def test_double_matches_spark(self, spark) -> None:
        values = [1.0, 0.5, 1e-10, 123456789.123, -0.0, 0.0, 100.0, 999999.9, 10000000.0, 0.001, 0.0001, -5.5, None]
        rows = spark.createDataFrame([(v,) for v in values], 'v DOUBLE').selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
        got = df.select(spark_cast_to_string('v', pl.Float64).alias('v'))['v'].to_list()
        assert got == expected

    def test_list_string_matches_spark(self, spark) -> None:
        values = [['1', '2'], [], None, ['a', None]]
        rows = spark.createDataFrame([(v,) for v in values], 'v ARRAY<STRING>').selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': pl.List(pl.String)})
        got = df.select(spark_cast_to_string('v', pl.List(pl.String)).alias('v'))['v'].to_list()
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
        got = df.select(spark_cast_to_string('v', dtype).alias('v'))['v'].to_list()
        assert got == expected

    def test_double_is_exact_immediately_below_the_1_7e16_cutoff(self, spark) -> None:
        # Guards the measured cutoff itself: _java_double_to_string raises at |x| >= 1.7e16, a
        # value chosen with a full 1e15 of headroom below the smallest real mismatch found
        # (1.801526131658083e16, task-8-report.md). A dense 300-value sample seeded for
        # determinism, concentrated right below the cutoff -- if the cutoff is ever widened
        # without re-measuring, a sample this dense just inside [1.6e16, 1.7e16) is likely to
        # catch a real divergence rather than let it slip through untested.
        rng = random.Random(20260814)
        values = [rng.uniform(1.6, 1.699999) * 1e16 for _ in range(300)]
        rows = spark.createDataFrame([(v,) for v in values], 'v DOUBLE').selectExpr('CAST(v AS STRING) as s')
        expected = [r['s'] for r in rows.collect()]
        df = pl.DataFrame({'v': values}, schema={'v': pl.Float64})
        got = df.select(spark_cast_to_string('v', pl.Float64).alias('v'))['v'].to_list()
        assert got == expected
