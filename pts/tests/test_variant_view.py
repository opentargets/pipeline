"""Tests for the variant view transformer, which replaces the `variants.sql` serving script."""

from pathlib import Path

import polars as pl
import pytest
from ot_datamodels.outputs.credible_set_variant import CredibleSetVariant
from ot_datamodels.outputs.pharmacogenomics import Pharmacogenomics
from ot_datamodels.outputs.variant import Variant
from ot_datamodels.polars import to_polars_schema
from ot_datamodels.views.variant_view import VariantView

from pts.schemas.dataset.validate import validate_dataset
from pts.task_outputs import OutputValidation
from pts.transformers import variant_view as module
from pts.transformers.variant_view import build_variant_view, variant_ids, variant_view
from tests.credible_set_fixtures import published_credible_set_variant

VARIANT = to_polars_schema(Variant)
PHARMACOGENOMICS = to_polars_schema(Pharmacogenomics)
V1, V2, V3 = '1_100_A_G', '1_200_C_T', '2_300_G_A'
SET_A, SET_B, SET_C = 'a' * 32, 'b' * 32, 'c' * 32


def _variants(*ids: str) -> pl.DataFrame:
    return pl.DataFrame([Variant.example | {'variantId': i} for i in ids], schema=VARIANT)


def _members(*pairs: tuple[str, str]) -> pl.DataFrame:
    rows = ({'credibleSetId': s, 'variantId': v, 'posteriorProbability': 0.5} for s, v in pairs)
    return published_credible_set_variant(*rows)


def _pharmacogenomics(*pairs: tuple[int, str | None]) -> pl.DataFrame:
    rows = [Pharmacogenomics.example | {'pharmacogenomicsId': i, 'variantId': v} for i, v in pairs]
    return pl.DataFrame(rows, schema=PHARMACOGENOMICS)


MEMBERS = _members((SET_C, V1), (SET_A, V1), (SET_B, V2), (SET_A, '9_9_A_C'))
PGX = _pharmacogenomics((7, V1), (3, V1), (5, V3), (9, None), (11, '9_9_A_C'))


def _ids(members: pl.DataFrame = MEMBERS, pgx: pl.DataFrame = PGX) -> pl.DataFrame:
    return variant_ids(
        members.lazy().select('credibleSetId', 'variantId'), pgx.lazy().select('pharmacogenomicsId', 'variantId')
    )


def test_id_lists_sorted_per_variant_empty_for_none():
    """Ids of a variant not in `variant` and pharmacogenomics rows without a variant are ignored; rows keep
    variant order."""
    view = build_variant_view(_variants(V3, V1, V2, '3_1_T_C'), _ids())

    assert view.select('variantId', 'credibleSetIds', 'pharmacogenomicsIds').rows() == [
        (V3, [], [5]),
        (V1, [SET_A, SET_C], [3, 7]),
        (V2, [SET_B], []),
        ('3_1_T_C', [], []),
    ]


def test_variant_columns_pass_through():
    variant = _variants(V1, V2)
    view = build_variant_view(variant, _ids())

    assert view.schema == to_polars_schema(VariantView)
    assert view.drop('credibleSetIds', 'pharmacogenomicsIds').equals(variant)


def test_empty_id_sources_give_empty_lists():
    view = build_variant_view(_variants(V1, V2), _ids(_members(), _pharmacogenomics()))

    assert view.schema == to_polars_schema(VariantView)
    assert view.select('credibleSetIds', 'pharmacogenomicsIds').rows() == [([], []), ([], [])]


def _source(root: Path, variant_parts: list[pl.DataFrame], members: pl.DataFrame, pgx: pl.DataFrame) -> dict:
    source = {}
    for name, parts in (('variant', variant_parts), ('credible_set_variant', [members]), ('pharmacogenomics', [pgx])):
        (root / name).mkdir(parents=True)
        for i, part in enumerate(parts):
            part.write_parquet(root / name / f'part-{i}.parquet')
        source[name] = str(root / name)
    return source


@pytest.mark.parametrize('concurrency', [1, 3])
def test_one_part_per_variant_part(tmp_path: Path, concurrency: int):
    parts = [_variants(V3, V1), _variants(), _variants(V2)]
    source = _source(tmp_path / 'src', parts, MEMBERS, PGX)
    destination = {'view': str(tmp_path / 'out' / 'view')}

    variant_view(source, destination, {'concurrency': concurrency}, None)

    written = sorted((tmp_path / 'out' / 'view').glob('*.parquet'))
    assert [p.name for p in written] == ['part-00000.parquet', 'part-00001.parquet', 'part-00002.parquet']
    assert [pl.read_parquet(p)['variantId'].to_list() for p in written] == [[V3, V1], [], [V2]]
    validate_dataset(VariantView, destination['view'])


def test_refuses_an_existing_destination(tmp_path: Path):
    source = _source(tmp_path / 'src', [_variants(V1)], MEMBERS, PGX)
    (tmp_path / 'out').mkdir()

    with pytest.raises(ValueError, match='exist'):
        variant_view(source, {'view': str(tmp_path / 'out')}, {}, None)


def test_writes_its_model(tmp_path: Path):
    variant_id = Variant.example['variantId']
    members = published_credible_set_variant(CredibleSetVariant.example | {'variantId': variant_id})
    pgx = pl.DataFrame([Pharmacogenomics.example | {'variantId': variant_id}], schema=PHARMACOGENOMICS)
    source = _source(tmp_path / 'src', [pl.DataFrame([Variant.example], schema=VARIANT)], members, pgx)
    destination = {'view': str(tmp_path / 'out' / 'view')}

    variant_view(source, destination, None, None)

    assert OutputValidation.of(variant_view, module).outputs == {'view': VariantView}
    validate_dataset(VariantView, destination['view'])
    written = pl.read_parquet(destination['view'])
    assert written.schema == to_polars_schema(VariantView)
    assert written.select('credibleSetIds', 'pharmacogenomicsIds').row(0) == (
        [CredibleSetVariant.example['credibleSetId']],
        [Pharmacogenomics.example['pharmacogenomicsId']],
    )
