"""The credible set stages run end to end on real new-schema credible sets of three sources."""

import importlib
from pathlib import Path
from typing import Any
from unittest.mock import Mock

import polars as pl
import pytest
from otter.config.model import Config
from polars.io.plugins import register_io_source

from pts.schemas.credible_set import (
    credible_set_schema,
    credible_set_variant_schema,
    validated_credible_set_schema,
)
from pts.schemas.study import study_schema
from pts.schemas.target import GENOMIC_LOCATION
from pts.schemas.variant import variant_schema
from pts.transformers.credible_set.annotation import annotation
from pts.transformers.credible_set.confidence import CredibleSetConfidence
from pts.transformers.credible_set.flags import CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import set_qc
from pts.transformers.credible_set.validation import validation
from pts.transformers.credible_set.variant_qc import variant_qc
from pts.transformers.study import QualityFlag as StudyQualityFlag
from pts.transformers.utils.dataset import scan_dataset
from tests.credible_set_fixtures import (
    ALT_CONTIG_SET,
    ALT_CONTIG_VARIANTS,
    CLEAN_SET,
    CLEAN_VARIANTS,
    MHC_SET,
    MHC_VARIANTS,
)

annotation_module = importlib.import_module('pts.transformers.credible_set.annotation')
"""The annotation module itself: the package exports its `annotation` function under the same name."""

UNREPORTED = {**CLEAN_VARIANTS[-1], 'effectAlleleFrequency': None}
"""A clean variant whose source frequency is dropped, so that annotation takes it from gnomAD."""
FIN_ADJ = 0.42
"""The gnomAD Finnish frequency of the unreported variant in the variant index."""

SOURCES = {
    'gcca': ([ALT_CONTIG_SET], ALT_CONTIG_VARIANTS),
    'gcss_pics': ([MHC_SET], MHC_VARIANTS),
    'finngen_gwas': ([CLEAN_SET], [*CLEAN_VARIANTS[:-1], UNREPORTED]),
}
"""Each fixture set with its variants, under the source it was staged from."""


def _write_indexes(root: Path) -> None:
    """The indexes annotation reads.

    Every study is a Finnish case-control GWAS of 500,000 samples and 20,000 cases. The variant index
    has the unreported variant's frequency, and the target and disease indexes one entry each.
    """
    studies = [
        {
            name: {
                'studyId': sets[0]['studyId'],
                'studyType': 'gwas',
                'qualityControls': [StudyQualityFlag.CASE_CONTROL_STUDY_DESIGN.value],
                'nSamples': 500_000,
                'nCases': 20_000,
                'ldPopulationStructure': [{'ldPopulation': 'fin', 'relativeSampleSize': 1.0}],
            }.get(name)
            for name in study_schema
        }
        for sets, _ in SOURCES.values()
    ]
    _write(studies, study_schema, root / 'output/study')
    frequencies = [{'populationName': 'fin_adj', 'alleleFrequency': FIN_ADJ}]
    variant = {'variantId': UNREPORTED['variantId'], 'alleleFrequencies': frequencies}
    _write([{name: variant.get(name) for name in variant_schema}], variant_schema, root / 'output/variant')
    location = {'chromosome': '1', 'start': 1, 'end': 2, 'strand': 1}
    _write([{'id': 'ENSG0', 'tss': 1, 'genomicLocation': location}], pl.Schema(TARGETS), root / 'output/target')
    _write([{'id': 'EFO_0', 'therapeuticAreas': []}], pl.Schema(DISEASES), root / 'output/disease')


TARGETS = {'id': pl.String, 'tss': pl.Int64, 'genomicLocation': GENOMIC_LOCATION}
DISEASES = {'id': pl.String, 'therapeuticAreas': pl.List(pl.String)}


def _write(rows: list[dict[str, Any]], schema: pl.Schema, path: Path) -> None:
    path.mkdir(parents=True)
    pl.DataFrame(rows, schema=schema).write_parquet(path / 'part-0.parquet')


def _read(path: Path) -> pl.DataFrame:
    return pl.read_parquet(list(path.glob('*.parquet')))


def _run_stages(root: Path) -> None:
    """Run the stages as `pts/config.yaml` chains them: QC per source, validation and annotation over all."""
    config = Mock(spec=Config, work_path=str(root), release_uri=None)

    def path(name: str) -> str:
        return str(root / name)

    _write_indexes(root)
    for source, (sets, variants) in SOURCES.items():
        _write(sets, credible_set_schema, root / f'input/gwas/{source}/credible_set')
        _write(variants, credible_set_variant_schema, root / f'input/gwas/{source}/credible_set_variant')
        variant_qc(
            source={
                'credible_set_variant': path(f'input/gwas/{source}/credible_set_variant'),
            },
            destination=path(f'intermediate/qc/gwas/{source}/credible_set_variant'),
            settings={},
            config=config,
        )
        set_qc(
            source={
                'credible_set': path(f'input/gwas/{source}/credible_set'),
                'study': path('output/study'),
            },
            destination=path(f'intermediate/qc/gwas/{source}/credible_set'),
            settings={},
            config=config,
        )
    validation(
        source={
            'credible_set': [path(f'intermediate/qc/gwas/{source}/credible_set') for source in SOURCES],
            'credible_set_variant': [path(f'intermediate/qc/gwas/{source}/credible_set_variant') for source in SOURCES],
        },
        destination={
            'valid_set': path('intermediate/validated/credible_set'),
            'valid_variant': path('intermediate/validated/credible_set_variant'),
            'excluded_set': path('excluded/credible_set'),
            'excluded_variant': path('excluded/credible_set_variant'),
        },
        settings={'cs_invalid_qc_reasons': ['IN_MHC'], 'csv_set_invalid_qc_reasons': ['INVALID_VARIANT_ID']},
        config=config,
    )
    annotation(
        source={
            'credible_set': path('intermediate/validated/credible_set'),
            'credible_set_variant': path('intermediate/validated/credible_set_variant'),
            'study': path('output/study'),
            'variant': path('output/variant'),
            'target': path('output/target'),
            'disease': path('output/disease'),
        },
        destination={
            'credible_set': path('output/credible_set'),
            'credible_set_variant': path('output/credible_set_variant'),
        },
        settings={},
        config=config,
    )


def test_invalid_and_mhc_sets_are_excluded_with_their_variants(tmp_path: Path) -> None:
    _run_stages(tmp_path)
    excluded_sets = {row['id']: row['qualityControls'] for row in _read(tmp_path / 'excluded/credible_set').to_dicts()}
    excluded_variants = _read(tmp_path / 'excluded/credible_set_variant')

    assert excluded_sets.keys() == {ALT_CONTIG_SET['id'], MHC_SET['id']}
    assert CredibleSetFlag.INVALID_VARIANT_IN_SET in excluded_sets[ALT_CONTIG_SET['id']]
    assert CredibleSetFlag.IN_MHC in excluded_sets[MHC_SET['id']]
    assert sorted(excluded_variants['variantId']) == sorted(v['variantId'] for v in ALT_CONTIG_VARIANTS + MHC_VARIANTS)
    assert excluded_variants['qualityControls'].list.contains(CredibleSetVariantFlag.CREDIBLE_SET_EXCLUDED).all()
    invalid = excluded_variants.filter(
        pl.col('qualityControls').list.contains(CredibleSetVariantFlag.INVALID_VARIANT_ID)
    )
    assert invalid['variantId'].to_list() == ['14_KI270846v1_alt_497149_C_T']


def test_the_valid_sets_of_every_source_are_published_together(tmp_path: Path) -> None:
    _run_stages(tmp_path)
    published_sets = _read(tmp_path / 'output/credible_set')
    published_variants = _read(tmp_path / 'output/credible_set_variant')

    assert published_sets['id'].to_list() == [CLEAN_SET['id']]
    assert published_sets['confidence'].to_list() == [CredibleSetConfidence.FINEMAPPED_IN_SAMPLE_LD]
    assert sorted(published_variants['variantId']) == sorted(v['variantId'] for v in CLEAN_VARIANTS)


def test_published_variants_take_the_source_frequency_or_else_the_gnomad_one(tmp_path: Path) -> None:
    _run_stages(tmp_path)
    published = {row['variantId']: row for row in _read(tmp_path / 'output/credible_set_variant').to_dicts()}

    for reported in CLEAN_VARIANTS[:-1]:
        row = published[reported['variantId']]
        assert row['effectAlleleFrequency'] == pytest.approx(reported['effectAlleleFrequency'])
        assert CredibleSetVariantFlag.EAF_FROM_SOURCE in row['qualityControls']
    row = published[UNREPORTED['variantId']]
    assert row['effectAlleleFrequency'] == pytest.approx(FIN_ADJ)
    assert CredibleSetVariantFlag.EAF_FROM_GNOMAD in row['qualityControls']


def test_published_variants_are_rescaled_on_the_log_odds_with_their_conditional_direction(tmp_path: Path) -> None:
    _run_stages(tmp_path)
    published = _read(tmp_path / 'output/credible_set_variant')

    assert published['rescaledDirection'].to_list() == [1] * len(CLEAN_VARIANTS)
    assert (published['rescaledAbsBeta'] > 0).all()
    assert published['qualityControls'].list.contains(CredibleSetVariantFlag.RESCALED_FROM_CONDITIONAL).all()


def test_every_stage_writes_the_new_schemas(tmp_path: Path) -> None:
    _run_stages(tmp_path)

    for source in SOURCES:
        assert _read(tmp_path / f'intermediate/qc/gwas/{source}/credible_set').schema == credible_set_schema
        assert (
            _read(tmp_path / f'intermediate/qc/gwas/{source}/credible_set_variant').schema
            == credible_set_variant_schema
        )
    assert _read(tmp_path / 'excluded/credible_set').schema == credible_set_schema
    for name in ('intermediate/validated/credible_set', 'output/credible_set'):
        assert _read(tmp_path / name).schema == validated_credible_set_schema, name
    for name in (
        'intermediate/validated/credible_set_variant',
        'excluded/credible_set_variant',
        'output/credible_set_variant',
    ):
        assert _read(tmp_path / name).schema == credible_set_variant_schema, name


def test_annotation_reads_each_input_dataset_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Every branch of the annotation plan shares one read of each dataset, instead of reading it again from storage."""
    read: dict[str, int] = {}

    def counting_scan(path: str, **options: Any) -> pl.LazyFrame:
        frame = scan_dataset(path, **options).collect()

        def source(with_columns, predicate, n_rows, batch_size):
            read[Path(path).name] = read.get(Path(path).name, 0) + 1
            yield frame

        return register_io_source(source, schema=frame.schema, is_pure=True)

    monkeypatch.setattr(annotation_module, 'scan_dataset', counting_scan)
    _run_stages(tmp_path)
    assert read == dict.fromkeys(('credible_set', 'credible_set_variant', 'study', 'target', 'disease', 'variant'), 1)
