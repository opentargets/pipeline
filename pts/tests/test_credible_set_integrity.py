"""Validation: integrity checks of the written datasets, and the output sort order."""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set import validation as validation_module
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import (
    integrity_checks,
    integrity_errors,
    sort_sets,
    sort_variants,
    validation,
    validation_check,
)
from pts.transformers.utils.dataset import scan_dataset

MHC = CredibleSetFlag.IN_MHC.value
EXCLUDED = CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED.value


def _sets(*rows: tuple) -> pl.LazyFrame:
    """Sets of (id, leadVariantId, studyType, qualityControls)."""
    return pl.LazyFrame(
        [
            {
                name: {'id': i, 'chromosome': '1', 'leadVariantId': lead, 'studyType': t, 'qualityControls': qc}.get(
                    name
                )
                for name in credible_set_schema
            }
            for i, lead, t, qc in rows
        ],
        schema=credible_set_schema,
    )


def _variants(*rows: tuple) -> pl.LazyFrame:
    """Variants of (credibleSetId, variantId, qualityControls)."""
    return pl.LazyFrame(
        [
            {
                name: {'credibleSetId': cs, 'variantId': v, 'qualityControls': qc}.get(name)
                for name in credible_set_variant_schema
            }
            for cs, v, qc in rows
        ],
        schema=credible_set_variant_schema,
    )


VALID_SET = ('a', '1_1_A_T', 'gwas', [])
EXCLUDED_SET = ('b', '1_5_A_T', 'gwas', [MHC])
VALID_VARIANTS = [('a', '1_1_A_T', []), ('a', '1_2_A_T', [])]
EXCLUDED_VARIANTS = [('b', '1_5_A_T', [EXCLUDED])]


def _errors(
    valid_set: pl.LazyFrame | None = None,
    valid_variant: pl.LazyFrame | None = None,
    excluded_set: pl.LazyFrame | None = None,
    excluded_variant: pl.LazyFrame | None = None,
) -> list[str]:
    """The integrity errors of one source, whose inputs are VALID_SET, EXCLUDED_SET and their variants."""
    checks = integrity_checks(
        'qc/gwas/s',
        _sets(VALID_SET, EXCLUDED_SET),
        _variants(*VALID_VARIANTS, *EXCLUDED_VARIANTS),
        valid_set if valid_set is not None else _sets(VALID_SET),
        valid_variant if valid_variant is not None else _variants(*VALID_VARIANTS),
        excluded_set if excluded_set is not None else _sets(EXCLUDED_SET),
        excluded_variant if excluded_variant is not None else _variants(*EXCLUDED_VARIANTS),
    )
    return integrity_errors(checks.collect())


def test_consistent_outputs_have_no_integrity_error() -> None:
    assert _errors() == []


def test_a_lost_set_is_an_error() -> None:
    [error] = _errors(excluded_set=_sets())
    assert 'qc/gwas/s' in error and 'sets' in error


def test_a_lost_variant_is_an_error() -> None:
    [error] = _errors(valid_variant=_variants(VALID_VARIANTS[0]), excluded_variant=_variants(*EXCLUDED_VARIANTS))
    assert 'qc/gwas/s' in error and 'variants' in error


def _write_source(root: Path) -> tuple[Path, Path]:
    """One QC'd source of VALID_SET and EXCLUDED_SET, each with its lead row."""
    sets = root / 'qc/gwas/s/credible_set'
    variants = root / 'qc/gwas/s/credible_set_variant'
    sets.mkdir(parents=True)
    variants.mkdir(parents=True)
    _sets(VALID_SET, EXCLUDED_SET).collect().write_parquet(sets / 'part-0.parquet')
    _variants(('a', '1_1_A_T', []), ('b', '1_5_A_T', [])).collect().write_parquet(variants / 'part-0.parquet')
    return sets, variants


def _validation(root: Path, sets: Path, variants: Path) -> None:
    validation(
        source={'credible_set': [str(sets)], 'credible_set_variant': [str(variants)]},
        destination={name: str(root / 'out' / name) for name in DESTINATIONS},
        settings={'cs_invalid_qc_reasons': ['IN_MHC']},
        config=Mock(spec=Config, work_path=str(root), release_uri=None),
    )


DESTINATIONS = ('valid_set', 'valid_variant', 'excluded_set', 'excluded_variant')


def _check(root: Path, sets: Path, variants: Path) -> None:
    validation_check(
        source={'credible_set': [str(sets)], 'credible_set_variant': [str(variants)]},
        destination={name: str(root / 'out' / name) for name in DESTINATIONS},
        settings={},
        config=Mock(spec=Config, work_path=str(root), release_uri=None),
    )


def test_the_written_datasets_pass_the_check(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    _validation(tmp_path, *source)
    _check(tmp_path, *source)


def test_the_check_fails_when_a_written_row_is_lost(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    _validation(tmp_path, *source)
    for part in (tmp_path / 'out/valid_variant').iterdir():
        part.unlink()
    _variants().collect().write_parquet(tmp_path / 'out/valid_variant/part-0.parquet')
    with pytest.raises(ValueError, match='2 input variants != 0 valid \\+ 1 excluded variants'):
        _check(tmp_path, *source)


def test_valid_variants_are_written_in_genome_order(tmp_path: Path) -> None:
    sets, variants = _write_source(tmp_path)
    _sets(('a', '1_1_A_T', 'gwas', [])).collect().write_parquet(sets / 'part-0.parquet')
    written = (
        ('a', 'X_3_A_T', []),
        ('a', '10_2_A_T', []),
        ('a', '1_1_A_T', []),
        ('a', '2_7_A_T', []),
        ('a', '1_10_A_T', []),
    )
    _variants(*written).collect().write_parquet(variants / 'part-0.parquet')
    _validation(tmp_path, sets, variants)
    valid = scan_dataset(str(tmp_path / 'out/valid_variant')).collect()
    assert valid['variantId'].to_list() == ['1_1_A_T', '1_10_A_T', '2_7_A_T', '10_2_A_T', 'X_3_A_T']
    assert valid.schema == credible_set_variant_schema


def test_sets_are_sorted_by_chromosome_rank_then_locus_then_lead() -> None:
    rows = [
        ('10', 5, 5, 'c'),
        ('2', 5, 5, 'b'),
        ('X', 1, 1, 'x'),
        ('1', 9, 9, 'z'),
        ('1', 1, 9, 'b'),
        ('1', 1, 9, 'a'),
        ('MT', 1, 1, 'm'),
        ('1', 1, 2, 'y'),
    ]
    sets = pl.LazyFrame(
        [{'chromosome': c, 'locusStart': s, 'locusEnd': e, 'leadVariantId': lead} for c, s, e, lead in rows],
        schema={'chromosome': pl.String, 'locusStart': pl.Int32, 'locusEnd': pl.Int32, 'leadVariantId': pl.String},
    )
    ordered = sort_sets(sets).collect()
    assert list(zip(ordered['chromosome'], ordered['leadVariantId'], strict=True)) == [
        ('1', 'y'),
        ('1', 'a'),
        ('1', 'b'),
        ('1', 'z'),
        ('2', 'b'),
        ('10', 'c'),
        ('X', 'x'),
        ('MT', 'm'),
    ]


def test_variants_are_sorted_by_chromosome_position_alleles_then_credible_set() -> None:
    variants = pl.LazyFrame({
        'credibleSetId': ['b', 'a', 'a', 'a', 'a', 'a', 'a', 'a'],
        'variantId': [
            '1_100_A_T',
            'MT_1_A_T',
            '1_100_A_T',
            '10_1_A_T',
            '1_100_A_C',
            '1_20_G_A',
            '2_1_A_T',
            '1_100_AT_T',
        ],
    })
    assert sort_variants(variants).collect().rows() == [
        ('a', '1_20_G_A'),
        ('a', '1_100_A_C'),
        ('a', '1_100_A_T'),
        ('b', '1_100_A_T'),
        ('a', '1_100_AT_T'),
        ('a', '2_1_A_T'),
        ('a', '10_1_A_T'),
        ('a', 'MT_1_A_T'),
    ]


def test_validation_scans_each_source_dataset_once(tmp_path: Path, monkeypatch) -> None:
    scanned: list[str] = []

    def counting_scan(path: str, **options):
        scanned.append(path)
        return scan_dataset(path, **options)

    monkeypatch.setattr(validation_module, 'scan_dataset', counting_scan)
    sets, variants = _write_source(tmp_path)
    _validation(tmp_path, sets, variants)
    assert sorted(scanned) == sorted([str(sets), str(variants)])
