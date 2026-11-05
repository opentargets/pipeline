"""Validation: integrity checks of the written datasets, and a write without a sort."""

from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.io.plugins import register_io_source

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set import validation as validation_module
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.validation import (
    integrity_checks,
    integrity_errors,
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


def test_validation_writes_every_dataset_without_sorting(tmp_path: Path, monkeypatch) -> None:
    """A global sort cannot stream: over every credible set variant it spills past the vm's disk budget.

    The only sort is the ranking of duplicated ids, over the valid sets.
    """
    written: list[pl.LazyFrame] = []
    collect_all = pl.collect_all

    def capturing_collect_all(frames: list[pl.LazyFrame], **options) -> list[pl.DataFrame]:
        written.extend(frames)
        return collect_all(frames, **options)

    monkeypatch.setattr(validation_module.pl, 'collect_all', capturing_collect_all)
    _validation(tmp_path, *_write_source(tmp_path))
    assert len(written) == len(DESTINATIONS)
    ranking = 'SORT BY [nulls_last: [true, true, true, true]] [col("id"), col("_size"), col("_source"), col("_hash")]'
    for frame in written:
        sort = [line.strip() for line in frame.explain().splitlines() if 'SORT' in line]
        assert all(line == ranking for line in sort), sort


def test_valid_variants_keep_the_credible_set_variant_schema(tmp_path: Path) -> None:
    _validation(tmp_path, *_write_source(tmp_path))
    assert scan_dataset(str(tmp_path / 'out/valid_variant')).collect().schema == credible_set_variant_schema


def test_validation_scans_each_source_dataset_once(tmp_path: Path, monkeypatch) -> None:
    scanned: list[str] = []

    def counting_scan(path: str, **options):
        scanned.append(path)
        return scan_dataset(path, **options)

    monkeypatch.setattr(validation_module, 'scan_dataset', counting_scan)
    sets, variants = _write_source(tmp_path)
    _validation(tmp_path, sets, variants)
    assert sorted(scanned) == sorted([str(sets), str(variants)])


def test_validation_reads_each_source_dataset_once(tmp_path: Path, monkeypatch) -> None:
    """Every branch of the plan shares one read of each dataset, instead of reading it again from storage."""
    read: list[str] = []

    def counting_scan(path: str, **options) -> pl.LazyFrame:
        frame = scan_dataset(path, **options).collect()

        def source(with_columns, predicate, n_rows, batch_size):
            read.append(path)
            yield frame

        return register_io_source(source, schema=frame.schema, is_pure=True)

    monkeypatch.setattr(validation_module, 'scan_dataset', counting_scan)
    sets, variants = _write_source(tmp_path)
    _validation(tmp_path, sets, variants)
    assert sorted(read) == sorted([str(sets), str(variants)])
