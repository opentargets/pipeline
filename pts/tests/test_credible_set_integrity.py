"""Validation: integrity checks of the written datasets, and a write without a sort."""

import importlib
from pathlib import Path
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.io.plugins import register_io_source

from pts.schemas.credible_set import credible_set_schema, credible_set_variant_schema
from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.validation import validation, validation_check
from pts.transformers.utils import dataset as dataset_module
from pts.transformers.utils.dataset import scan_dataset

validation_module = importlib.import_module('pts.transformers.credible_set.validation')
"""The validation module itself: the package exports its `validation` function under the same name."""

MHC = CredibleSetFlag.IN_MHC.value


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


def _empty(path: Path) -> None:
    """Replace the written dataset at `path` with an empty one of the same schema."""
    empty = pl.read_parquet(path / '*.parquet').clear()
    for part in path.iterdir():
        part.unlink()
    empty.write_parquet(path / 'part-0.parquet')


def test_the_check_fails_when_a_written_set_is_lost(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    _validation(tmp_path, *source)
    _empty(tmp_path / 'out/excluded_set')
    with pytest.raises(ValueError, match='2 input sets != 1 valid \\+ 0 excluded sets'):
        _check(tmp_path, *source)


def test_the_check_fails_when_a_written_row_is_lost(tmp_path: Path) -> None:
    source = _write_source(tmp_path)
    _validation(tmp_path, *source)
    _empty(tmp_path / 'out/valid_variant')
    with pytest.raises(ValueError, match='2 input variants != 0 valid \\+ 1 excluded variants'):
        _check(tmp_path, *source)


def test_validation_ranks_duplicated_ids_once_and_writes_without_sorting(tmp_path: Path, monkeypatch) -> None:
    """A global sort cannot stream: over every credible set variant it spills past the vm's disk budget.

    The only sort is the ranking of duplicated ids over the valid sets of every source, collected
    once; the per-source collects and the writes that follow do not sort.
    """
    collected: list[pl.LazyFrame] = []
    sorted_plan: list[str] = []
    collect_all, collect = pl.collect_all, pl.LazyFrame.collect

    def capturing_collect_all(frames: list[pl.LazyFrame], **options) -> list[pl.DataFrame]:
        collected.extend(frames)
        return collect_all(frames, **options)

    def capturing_collect(frame: pl.LazyFrame, *args, **options) -> pl.DataFrame:
        sorted_plan.extend(line.strip() for line in frame.explain().splitlines() if 'SORT' in line)
        return collect(frame, *args, **options)

    monkeypatch.setattr(validation_module.pl, 'collect_all', capturing_collect_all)
    monkeypatch.setattr(pl.LazyFrame, 'collect', capturing_collect)
    _validation(tmp_path, *_write_source(tmp_path))
    ranking = 'SORT BY [nulls_last: [true, true, true, true]] [col("id"), col("_size"), col("_source"), col("_hash")]'
    assert sorted_plan == [ranking]
    assert not [line for frame in collected for line in frame.explain().splitlines() if 'SORT' in line]


def test_valid_variants_keep_the_credible_set_variant_schema(tmp_path: Path) -> None:
    _validation(tmp_path, *_write_source(tmp_path))
    assert scan_dataset(str(tmp_path / 'out/valid_variant')).collect().schema == credible_set_variant_schema


def test_validation_scans_each_source_dataset_once(tmp_path: Path, monkeypatch) -> None:
    scanned: list[str] = []

    def counting_scan(path: str, **options):
        scanned.append(path)
        return scan_dataset(path, **options)

    # validation reads through its own `scan_dataset`, and `scan_datasets` through the one it is defined beside
    monkeypatch.setattr(validation_module, 'scan_dataset', counting_scan)
    monkeypatch.setattr(dataset_module, 'scan_dataset', counting_scan)
    sets, variants = _write_source(tmp_path)
    _validation(tmp_path, sets, variants)
    assert sorted(scanned) == sorted([str(sets), str(variants)])


def test_validation_reads_the_sets_once_and_the_variants_three_times(tmp_path: Path, monkeypatch) -> None:
    """The sets are read once and kept in memory, for the cross-set QC and for validation.

    The variants are read once for the cross-set QC, once for the facts the sets are decided on, and
    once to split them.
    """
    read: list[str] = []

    def counting_scan(path: str, **options) -> pl.LazyFrame:
        frame = scan_dataset(path, **options).collect()

        def source(with_columns, predicate, n_rows, batch_size):
            read.append(path)
            yield frame

        return register_io_source(source, schema=frame.schema, is_pure=True)

    monkeypatch.setattr(validation_module, 'scan_dataset', counting_scan)
    monkeypatch.setattr(dataset_module, 'scan_dataset', counting_scan)
    sets, variants = _write_source(tmp_path)
    # a SuSiE and a PICS set, so the cross-set QC has members to read: with neither, polars skips them
    set_file = sets / 'part-0.parquet'
    pl.read_parquet(set_file).with_columns(
        pl.Series('fineMappingMethod', ['SuSiE', 'PICS']),
        pl.lit(1, pl.Int32).alias('locusStart'),
        pl.lit(10, pl.Int32).alias('locusEnd'),
    ).write_parquet(set_file)
    _validation(tmp_path, sets, variants)
    assert sorted(read) == sorted([str(sets), str(variants), str(variants), str(variants)])
