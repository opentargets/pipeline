"""Tests for the parametrised post-processing recipe (`pts.transformers.evidence.postprocess`).

The `Evidence` chain's own semantics are covered in `test_evidence_polars.py`. What is covered here
is the recipe as a reusable unit: that it is configured by parameters, that it neither reads nor
writes, and how it splits valid from invalid.

The json reading tests at the end cover the reader options the step passes, and the consequences
they carry into the chain.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.core import QC_COLUMN, Evidence, EvidenceFlags
from pts.transformers.evidence.postprocess import EvidencePostprocessor, ValidationLuts
from pts.transformers.utils.dataset import scan_dataset


def _read_json_evidence(path: str) -> pl.LazyFrame:
    """Read json evidence exactly as `evidence_postprocess` does.

    The step reads inline, so there is no function to call. This mirrors its json branch,
    `infer_schema_length` included: keep the two in step.
    """
    return scan_dataset(path, format='ndjson', infer_schema_length=None)


def _write_json_gz(path: Path, rows: list[dict]) -> str:
    """Write one gzipped newline delimited json file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = '\n'.join(json.dumps(row) for row in rows).encode()
    path.write_bytes(gzip.compress(payload))
    return str(path)


def _luts() -> ValidationLuts:
    """Minimal lookup tables resolving one target and one disease, built in memory.

    Constructed directly rather than through the `build_*_lut` readers: the recipe takes LUTs as
    values, so testing it needs no storage.
    """
    return ValidationLuts(
        disease=pl.DataFrame({'diseaseFromSourceMappedId': ['EFO_1'], 'diseaseId': ['EFO_1']}),
        target=pl.DataFrame(
            {'targetFromSourceId': ['ENSG1'], 'targetId': ['ENSG1'], 'biotype': ['protein_coding']}
        ),
        publication=pl.DataFrame({'publicationId': ['1'], 'publicationDate': ['2020-01-01']}),
    )


def _postprocessor(**overrides: object) -> EvidencePostprocessor:
    defaults: dict[str, object] = {
        'datasource_id': 'eva',
        'unique_fields': ['targetFromSourceId', 'diseaseFromSourceMappedId'],
        'score': pl.col('resourceScore'),
    }
    return EvidencePostprocessor(**(defaults | overrides))  # type: ignore[arg-type]


def _raw(**overrides: object) -> pl.LazyFrame:
    row: dict[str, object] = {
        'targetFromSourceId': 'ENSG1',
        'diseaseFromSourceMappedId': 'EFO_1',
        'datasourceId': 'eva',
        'resourceScore': 0.5,
    }
    return pl.LazyFrame([row | overrides])


def test_run_returns_lazy_frames_and_touches_no_storage() -> None:
    """The recipe is frame-to-frame: no paths in, no paths out, nothing collected.

    This is what lets a caller run it on evidence held in memory, with no intermediate file.
    """
    result = _postprocessor().run(_raw(), _luts())

    assert isinstance(result.valid, pl.LazyFrame)
    assert isinstance(result.invalid, pl.LazyFrame)


def test_a_fully_resolvable_row_comes_out_valid() -> None:
    result = _postprocessor().run(_raw(), _luts())

    valid = result.valid.collect()
    assert valid.height == 1
    assert result.invalid.collect().height == 0
    assert valid['targetId'].to_list() == ['ENSG1']
    assert valid['diseaseId'].to_list() == ['EFO_1']


def test_an_unresolvable_target_is_split_into_the_invalid_half() -> None:
    result = _postprocessor().run(_raw(targetFromSourceId='NOT_A_TARGET'), _luts())

    assert result.valid.collect().height == 0
    invalid = result.invalid.collect()
    assert invalid.height == 1
    assert EvidenceFlags.INVALID_TARGET in invalid[QC_COLUMN].item().to_list()


def test_the_score_expression_is_a_parameter_not_a_lookup() -> None:
    """Two postprocessors differing ONLY in their score expression produce different scores.

    Pins that the recipe never consults `EXPRESSIONS`, so a caller can supply its own.
    """
    luts, raw = _luts(), _raw()

    from_column = _postprocessor().run(raw, luts).valid.collect()
    constant = _postprocessor(score=pl.lit(0.25)).run(raw, luts).valid.collect()

    assert from_column['score'].to_list() == [0.5]
    assert constant['score'].to_list() == [0.25]


def test_excluded_biotypes_are_applied_when_given() -> None:
    luts = _luts()

    without = _postprocessor().run(_raw(), luts)
    with_exclusion = _postprocessor(excluded_biotypes=['protein_coding']).run(_raw(), luts)

    assert without.valid.collect().height == 1
    assert with_exclusion.valid.collect().height == 0
    flags = with_exclusion.invalid.collect()[QC_COLUMN].item().to_list()
    assert EvidenceFlags.INVALID_BIOTYPE in flags


def test_a_mismatched_datasource_is_dropped_entirely_not_flagged() -> None:
    """`datasource_id` FILTERS; it does not flag.

    It behaves differently from its neighbours: an unresolvable target or an excluded biotype lands
    in the invalid half and is still published as failed evidence, whereas a row whose
    `datasourceId` disagrees appears in neither half. A caller passing the wrong id therefore emits
    nothing at all, rather than a pile of failures.
    """
    result = _postprocessor(datasource_id='not_eva').run(_raw(), _luts())

    assert result.valid.collect().height == 0
    assert result.invalid.collect().height == 0


# --------------------------------------------------------------- reading json evidence


def test_json_evidence_reads_only_the_columns_the_source_carries(tmp_path: Path) -> None:
    """The frame carries the SOURCE's columns, not `evidence_schema`'s full field set.

    Pinning `schema=evidence_schema` at read time would materialise every field the schema knows
    about whether or not the source has it, filling the frame with all-null columns. Guards against
    a later change reintroducing such a pin.
    """
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1', 'resourceScore': 0.5}])

    frame = _read_json_evidence(path).collect()

    assert set(frame.columns) == {'targetFromSourceId', 'resourceScore'}
    assert len(frame.columns) < len(evidence_schema)


def test_json_evidence_leaves_known_dtypes_to_harmonisation(tmp_path: Path) -> None:
    """The read infers dtypes; `Evidence` casts the ones `evidence_schema` knows.

    `resourceScore: 1` infers as Int64 and is not corrected at read time, because harmonisation
    already covers every schema column. Pins both ends: inferred on the way in, schema dtype once
    the chain has seen it.
    """
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'resourceScore': 1}])

    assert _read_json_evidence(path).collect().schema['resourceScore'] == pl.Int64
    assert Evidence(_read_json_evidence(path)).lf.collect().schema['resourceScore'] == evidence_schema['resourceScore']


def test_json_evidence_finds_a_column_absent_from_a_bounded_sample(tmp_path: Path) -> None:
    """A column first appearing past polars' default inference window must not be dropped.

    Covered generically in `test_dataset.py`; repeated here because it is the reason the step
    passes `infer_schema_length=None` at all.
    """
    rows = [{'targetFromSourceId': f't{i}'} for i in range(200)]
    rows.append({'targetFromSourceId': 't200', 'resourceScore': 0.5})
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)

    frame = _read_json_evidence(path).collect()

    assert frame.filter(pl.col('targetFromSourceId') == 't200')['resourceScore'].item() == 0.5


def test_json_column_order_decides_the_uniqueness_survivor(tmp_path: Path) -> None:
    """Column order selects which of two duplicate rows is published.

    Harmonisation preserves column order, so the order columns arrive in reaches
    `validate_uniqueness`, whose content hash picks the surviving row among duplicates. Reading
    yields file order; sorting the same frame alphabetically picks the OTHER row.

    Both orderings are equally arbitrary, so neither is more correct -- but the choice is a
    content difference, not a cosmetic one, and it is pinned here so that a change to reader column
    order is visible rather than silent.

    Two rows share an `id` (same `keyField`) and differ only in `zCol`; `aCol` is constant so it
    cannot decide the ranking on its own.
    """
    rows = [
        {'keyField': 'same', 'zCol': 'a', 'aCol': 'a'},
        {'keyField': 'same', 'zCol': 'c', 'aCol': 'a'},
    ]
    lf = _read_json_evidence(_write_json_gz(tmp_path / 'evidence.json.gz', rows))
    assert lf.collect_schema().names() == ['keyField', 'zCol', 'aCol']

    survivor = Evidence(lf).assign_evidence_identifier(['keyField']).validate_uniqueness().lf.collect()
    flagged = {row['zCol']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in survivor.to_dicts()}
    assert flagged == {'a': True, 'c': False}  # 'c' survives in file order

    # Sorted into spark's alphabetical order, the SAME two rows pick the OTHER survivor.
    spark_order_lf = lf.select(sorted(lf.collect_schema().names()))
    spark_order = Evidence(spark_order_lf).assign_evidence_identifier(['keyField']).validate_uniqueness().lf.collect()
    spark_flagged = {row['zCol']: EvidenceFlags.DUPLICATED in row[QC_COLUMN] for row in spark_order.to_dicts()}
    assert spark_flagged == {'a': False, 'c': True}
