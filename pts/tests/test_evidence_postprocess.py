"""Tests for the parametrised post-processing recipe (`pts.transformers.evidence.postprocess`).

The `Evidence` chain's own semantics are covered end to end in `test_evidence_polars.py`. What is
new here is the recipe as a REUSABLE UNIT: that it is configured by parameters rather than by a
step's settings, that it neither reads nor writes, and that it splits valid from invalid.

These properties are what let the same recipe run inside today's `evidence_postprocess_*` step and,
later, inside a per-datasource module that generates its own evidence.
"""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.core import QC_COLUMN, Evidence, EvidenceFlags
from pts.transformers.evidence.expressions import DatasourceExpressions
from pts.transformers.evidence.postprocess import EvidencePostprocessor, ValidationLuts
from pts.transformers.utils.dataset import scan_dataset


def _read_json_evidence(path: str) -> pl.LazyFrame:
    """Read json evidence exactly as `evidence_postprocess` does.

    The step reads inline -- two `scan_dataset` calls picked by `evidence_format` -- so there is no
    function to call here. This mirrors its json branch, `infer_schema_length` included, which is
    the part with consequences worth pinning.
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
    values, so a test of the recipe needs no storage at all -- which is itself the property under
    test here.
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
        'expressions': DatasourceExpressions(score=pl.col('resourceScore')),
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
    """The recipe is pure frame-to-frame: no paths in, no paths out.

    This is the property that lets a future per-datasource module run the same recipe on evidence
    it generated in memory, never having written an intermediate file.
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
    """Two postprocessors differing ONLY in their expressions produce different scores.

    Pins the boundary the whole design rests on: this module never consults `EXPRESSIONS`, so a
    per-datasource module can supply its own expressions once the registry dissolves.
    """
    luts, raw = _luts(), _raw()

    from_column = _postprocessor().run(raw, luts).valid.collect()
    constant = _postprocessor(expressions=DatasourceExpressions(score=pl.lit(0.25))).run(raw, luts).valid.collect()

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

    Worth pinning explicitly because it is the one parameter that behaves differently from its
    neighbours: an unresolvable target or an excluded biotype lands in the invalid half and is
    still published as failed evidence, whereas a row whose `datasourceId` disagrees disappears
    from BOTH halves. A future per-datasource module passing the wrong id would silently emit
    nothing at all rather than a pile of failures.
    """
    result = _postprocessor(datasource_id='not_eva').run(_raw(), _luts())

    assert result.valid.collect().height == 0
    assert result.invalid.collect().height == 0


# --------------------------------------------------------------- reading json evidence


def test_json_evidence_reads_only_the_columns_the_source_carries(tmp_path: Path) -> None:
    """The frame must carry the SOURCE's columns, not evidence.json's full field set.

    Pinning `schema=evidence_schema` would MATERIALISE every field the schema knows about whether
    or not the source has it -- measured on real data, `reactome.json.gz`'s 12 real columns became
    109 that way, filled with spurious all-null columns spark never produced. Inference cannot do
    that, and this guards against a later change reintroducing a wholesale pin.
    """
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'targetFromSourceId': 't1', 'resourceScore': 0.5}])

    frame = _read_json_evidence(path).collect()

    assert set(frame.columns) == {'targetFromSourceId', 'resourceScore'}
    assert len(frame.columns) < len(evidence_schema)


def test_json_evidence_leaves_known_dtypes_to_harmonisation(tmp_path: Path) -> None:
    """The read infers dtypes; `Evidence` casts the ones `evidence_schema` knows.

    `resourceScore: 1` infers as Int64 and is deliberately NOT corrected at read time -- doing so
    would duplicate what harmonisation already does for every schema column, which is why the read
    no longer builds a schema at all. Pins both ends: inferred on the way in, schema dtype once the
    chain has seen it.
    """
    path = _write_json_gz(tmp_path / 'evidence.json.gz', [{'resourceScore': 1}])

    assert _read_json_evidence(path).collect().schema['resourceScore'] == pl.Int64
    assert Evidence(_read_json_evidence(path)).lf.collect().schema['resourceScore'] == evidence_schema['resourceScore']


def test_json_evidence_finds_a_column_absent_from_a_bounded_sample(tmp_path: Path) -> None:
    """A column first appearing past polars' default 100-row inference window must not be dropped.

    Covered generically in `test_dataset.py`; repeated here because it is the reason the step
    passes `infer_schema_length=None` at all, and a real source (`cosmic.json.gz`) measurably hit
    it.
    """
    rows = [{'targetFromSourceId': f't{i}'} for i in range(200)]
    rows.append({'targetFromSourceId': 't200', 'resourceScore': 0.5})
    path = _write_json_gz(tmp_path / 'evidence.json.gz', rows)

    frame = _read_json_evidence(path).collect()

    assert frame.filter(pl.col('targetFromSourceId') == 't200')['resourceScore'].item() == 0.5


def test_json_column_order_decides_the_uniqueness_survivor(tmp_path: Path) -> None:
    """Columns come out in FILE order, not spark's alphabetical -- and that picks a different row.

    Spark's json reader sorts inferred columns alphabetically; polars keeps file order. The read
    used to build a schema purely to force spark's order back; that was dropped deliberately, since
    matching the release byte for byte is not a requirement.

    This pins what the divergence costs. Two rows share an `id` (same `keyField`) and differ only
    in `zCol`; `aCol` is constant so it cannot decide the ranking. Harmonisation is
    order-preserving, so column order reaches `validate_uniqueness`, whose content hash decides
    which duplicate survives. In file order `zCol='c'` survives; in alphabetical order `zCol='a'`
    does. Same content, same id, DIFFERENT published row -- arbitrary either way (spark's is a hash
    order too), which is why it is an accepted trade rather than a defect, but a content difference
    and not a cosmetic one.

    The fixture was not hand-picked to "look plausible": a brute-force search over `zCol`/`aCol`
    values, run through the real `Evidence` pipeline, found the first pair whose survivor actually
    flips between the two orderings.
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
