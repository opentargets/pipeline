"""Tests for the variant search index."""

import polars as pl

from pts.transformers.search.helpers import LIST_STR
from pts.transformers.search.variant import build_variant_index, variant_labels

XREFS = pl.List(pl.Struct({'id': pl.String, 'source': pl.String}))

VARIANT_SCHEMA = {
    'variantId': pl.String,
    'rsIds': LIST_STR,
    'hgvsId': pl.String,
    'dbXrefs': XREFS,
    'chromosome': pl.String,
    'position': pl.Int32,
}


def _variants(rows):
    return pl.LazyFrame(rows, schema=VARIANT_SCHEMA)


def _one(**overrides):
    row = {
        'variantId': '1_100_A_G',
        'rsIds': ['rs1'],
        'hgvsId': 'hgvs1',
        'dbXrefs': [{'id': 'x1', 'source': 'clinvar'}],
        'chromosome': '1',
        'position': 100,
    }
    row.update(overrides)
    return row


def test_variant_index_carries_the_release_schema_and_a_multiplier_of_one() -> None:
    result = build_variant_index(_variants([_one()])).collect()

    assert result['id'][0] == '1_100_A_G'
    assert result['name'][0] == '1_100_A_G'
    assert result['entity'][0] == 'variant'
    assert result['category'][0].to_list() == ['variant']
    assert result['multiplier'][0] == 1.0
    assert result['description'][0] is None


def test_variant_index_terms_are_empty_not_null() -> None:
    result = build_variant_index(_variants([_one()])).collect()

    for column in ('terms', 'terms25', 'terms5'):
        assert result[column][0].to_list() == []


def test_variant_keywords_include_the_chr_prefixed_and_underscore_location_forms() -> None:
    keywords = build_variant_index(_variants([_one()])).collect()['keywords'][0].to_list()

    assert set(keywords) == {'1_100_A_G', 'chr1_100_A_G', 'hgvs1', 'x1', 'rs1', '1_100_', 'chr1_100_'}


def test_gnomad_xrefs_are_dropped_as_redundant_with_the_variant_id() -> None:
    row = _one(dbXrefs=[{'id': '1-100-A-G', 'source': 'gnomad'}, {'id': 'x1', 'source': 'clinvar'}])

    keywords = build_variant_index(_variants([row])).collect()['keywords'][0].to_list()

    assert '1-100-A-G' not in keywords
    assert 'x1' in keywords


def test_spark_null_sourced_xrefs_are_kept_because_the_comparison_is_null_safe() -> None:
    """`~source.eqNullSafe('gnomad')` is TRUE for a null source, so the xref survives. A plain
    `!=` would yield null and drop it."""
    row = _one(dbXrefs=[{'id': 'x9', 'source': None}])

    assert 'x9' in build_variant_index(_variants([row])).collect()['keywords'][0].to_list()


def test_spark_location_is_null_when_a_component_is_null_so_it_contributes_nothing() -> None:
    """Spark's `concat` propagates null, so a variant with no position has no location form at
    all rather than a literal 'None' in the index."""
    row = _one(chromosome=None)

    keywords = build_variant_index(_variants([row])).collect()['keywords'][0].to_list()

    assert not any(keyword.endswith('_100_') for keyword in keywords)


def test_variant_ngrams_hold_only_the_id_and_xrefs() -> None:
    ngrams = build_variant_index(_variants([_one()])).collect()['ngrams'][0].to_list()

    assert set(ngrams) == {'1_100_A_G', 'x1'}


def test_variant_labels_are_keyed_by_variant_and_exclude_location_forms() -> None:
    """The target index reuses these; they must NOT carry the location forms, which the
    pyspark job builds only for the variant index."""
    result = variant_labels(_variants([_one()])).collect()

    assert result.columns == ['variantId', 'variant_labels']
    assert set(result['variant_labels'][0].to_list()) == {'1_100_A_G', 'chr1_100_A_G', 'hgvs1', 'x1', 'rs1'}
