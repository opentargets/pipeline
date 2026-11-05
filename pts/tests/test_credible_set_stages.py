"""The credible set stages run end to end on real new-schema credible sets of three sources."""

from pathlib import Path
from typing import Any
from unittest.mock import Mock

import polars as pl
from otter.config.model import Config

from pts.schemas.credible_set import (
    credible_set_schema,
    credible_set_variant_schema,
    validated_credible_set_schema,
)
from pts.transformers.credible_set.annotation import annotation
from pts.transformers.credible_set.flags import CredibleSetConfidenceClasses, CredibleSetFlag, CredibleSetVariantFlag
from pts.transformers.credible_set.set_qc import set_qc
from pts.transformers.credible_set.validation import validation
from pts.transformers.credible_set.variant_qc import variant_qc
from tests.credible_set_fixtures import (
    ALT_CONTIG_SET,
    ALT_CONTIG_VARIANTS,
    CLEAN_SET,
    CLEAN_VARIANTS,
    MHC_SET,
    MHC_VARIANTS,
)

SOURCES = {
    'gcca': ([ALT_CONTIG_SET], ALT_CONTIG_VARIANTS),
    'gcss_pics': ([MHC_SET], MHC_VARIANTS),
    'finngen_gwas': ([CLEAN_SET], CLEAN_VARIANTS),
}
"""Each fixture set with its variants, under the source it was staged from."""


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

    for source, (sets, variants) in SOURCES.items():
        _write(sets, credible_set_schema, root / f'input/gwas/{source}/credible_set')
        _write(variants, credible_set_variant_schema, root / f'input/gwas/{source}/credible_set_variant')
        variant_qc(
            source={'credible_set_variant': path(f'input/gwas/{source}/credible_set_variant')},
            destination=path(f'intermediate/qc/gwas/{source}/credible_set_variant'),
            settings={},
            config=config,
        )
        set_qc(
            source={'credible_set': path(f'input/gwas/{source}/credible_set')},
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
    assert published_sets['confidence'].to_list() == [CredibleSetConfidenceClasses.FINEMAPPED_IN_SAMPLE_LD]
    assert sorted(published_variants['variantId']) == sorted(v['variantId'] for v in CLEAN_VARIANTS)


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
