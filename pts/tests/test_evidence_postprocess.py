"""Tests for the parametrised post-processing recipe (`pts.transformers.evidence.postprocess`).

The `Evidence` chain's own semantics are covered end to end in `test_evidence_polars.py`. What is
new here is the recipe as a REUSABLE UNIT: that it is configured by parameters rather than by a
step's settings, that it neither reads nor writes, and that it splits valid from invalid.

These properties are what let the same recipe run inside today's `evidence_postprocess_*` step and,
later, inside a per-datasource module that generates its own evidence.
"""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.core import QC_COLUMN, EvidenceFlags
from pts.transformers.evidence.expressions import DatasourceExpressions
from pts.transformers.evidence.postprocess import EvidencePostprocessor, ValidationLuts


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
