"""Tests for the target search index."""

import polars as pl

from pts.transformers.search.helpers import LIST_STR
from pts.transformers.search.target import build_target_index, variant_labels_by_target

XREFS = pl.List(pl.Struct({'id': pl.String, 'source': pl.String}))
LABELLED = pl.List(pl.Struct({'label': pl.String, 'source': pl.String}))
IDENTIFIED = pl.List(pl.Struct({'id': pl.String, 'source': pl.String}))
CONSEQUENCES = pl.List(
    pl.Struct({'targetId': pl.String, 'consequenceScore': pl.Float32, 'distanceFromFootprint': pl.Int64})
)

TARGET_SCHEMA = {
    'targetId': pl.String,
    'approvedSymbol': pl.String,
    'approvedName': pl.String,
    'biotype': pl.String,
    'synonyms': LABELLED,
    'proteinIds': IDENTIFIED,
    'dbXrefs': XREFS,
}

VARIANT_SCHEMA = {
    'variantId': pl.String,
    'rsIds': LIST_STR,
    'hgvsId': pl.String,
    'dbXrefs': XREFS,
    'transcriptConsequences': CONSEQUENCES,
}


def _target(**overrides):
    row = {
        'targetId': 'T1',
        'approvedSymbol': 'EGFR',
        'approvedName': 'receptor',
        'biotype': 'protein_coding',
        'synonyms': [{'label': 'ERBB1', 'source': 'x'}],
        'proteinIds': [{'id': 'P001', 'source': 'uniprot'}],
        'dbXrefs': [{'id': '1234', 'source': 'HGNC'}],
    }
    row.update(overrides)
    return row


def _variant(**overrides):
    row = {
        'variantId': '1_100_A_G',
        'rsIds': ['rs1'],
        'hgvsId': 'hgvs1',
        'dbXrefs': [{'id': 'x1', 'source': 'clinvar'}],
        'transcriptConsequences': [{'targetId': 'T1', 'consequenceScore': 1.0, 'distanceFromFootprint': 10}],
    }
    row.update(overrides)
    return row


def _inputs(**overrides):
    base = {
        'targets': pl.LazyFrame([_target()], schema=TARGET_SCHEMA),
        'associations': pl.LazyFrame(
            {'associationId': ['D1-T1'], 'targetId': ['T1'], 'diseaseId': ['D1'], 'score': [0.8]}
        ),
        'd_lut': pl.LazyFrame(
            {'diseaseId': ['D1'], 'disease_labels': [['asthma']]},
            schema={'diseaseId': pl.String, 'disease_labels': LIST_STR},
        ),
        'variants': pl.LazyFrame([_variant()], schema=VARIANT_SCHEMA),
    }
    base.update(overrides)
    return base


def test_target_index_carries_the_release_identity_columns() -> None:
    result = build_target_index(**_inputs()).collect()

    assert result['id'].to_list() == ['T1']
    assert result['name'].to_list() == ['EGFR']
    assert result['description'].to_list() == ['receptor']
    assert result['entity'].to_list() == ['target']
    assert result['category'][0].to_list() == ['protein_coding']


def test_target_keywords_include_the_hgnc_prefixed_identifier() -> None:
    keywords = build_target_index(**_inputs()).collect()['keywords'][0].to_list()

    assert set(keywords) == {'ERBB1', 'P001', 'receptor', 'EGFR', 'HGNC:1234', 'T1'}


def test_target_with_no_hgnc_xref_still_gets_a_document() -> None:
    """36,503 of 78,733 targets have no HGNC xref; they must not be dropped."""
    targets = pl.LazyFrame([_target(dbXrefs=[])], schema=TARGET_SCHEMA)

    result = build_target_index(**_inputs(targets=targets)).collect()

    assert result.height == 1
    assert 'EGFR' in result['keywords'][0].to_list()


def test_two_hgnc_xrefs_yield_one_document_carrying_both() -> None:
    """Latent in the pyspark job: `explode_outer` would emit TWO documents for this target.
    It never fires on current data (every target has 0 or 1 HGNC xref) so this is zero-diff,
    but the polars version must not be able to fan a target out."""
    targets = pl.LazyFrame(
        [_target(dbXrefs=[{'id': '1234', 'source': 'HGNC'}, {'id': '5678', 'source': 'HGNC'}])],
        schema=TARGET_SCHEMA,
    )

    result = build_target_index(**_inputs(targets=targets)).collect()

    assert result.height == 1
    assert {'HGNC:1234', 'HGNC:5678'} <= set(result['keywords'][0].to_list())


def test_target_terms_hold_disease_and_variant_labels() -> None:
    terms = build_target_index(**_inputs()).collect()['terms'][0].to_list()

    assert 'asthma' in terms
    assert '1_100_A_G' in terms


def test_target_terms_carry_no_drug_labels_matching_the_current_release() -> None:
    """Pins the CURRENT behaviour so the port is provably faithful. Task 12 changes this
    deliberately, in its own commit, with a measured delta."""
    result = build_target_index(**_inputs()).collect()

    assert result.height == 1


def test_variant_labels_by_target_keeps_only_the_top_ranked_variants() -> None:
    """With only two candidate variants for T1, both would trivially rank <= 5 -- padded with
    filler variants so the top-5 cutoff is actually exercised."""
    filler = [
        _variant(
            variantId=f'v_filler{i}',
            transcriptConsequences=[{'targetId': 'T1', 'consequenceScore': 1.0, 'distanceFromFootprint': i}],
        )
        for i in range(2, 6)
    ]
    variants = pl.LazyFrame(
        [
            _variant(
                variantId='v_near',
                transcriptConsequences=[{'targetId': 'T1', 'consequenceScore': 1.0, 'distanceFromFootprint': 1}],
            ),
            *filler,
            _variant(
                variantId='v_far',
                transcriptConsequences=[{'targetId': 'T1', 'consequenceScore': 1.0, 'distanceFromFootprint': 999}],
            ),
        ],
        schema=VARIANT_SCHEMA,
    )

    result = variant_labels_by_target(variants).collect()

    assert result.height == 1
    assert 'v_near' in result['variant_labels_5'][0].to_list()
    assert 'v_far' not in result['variant_labels_5'][0].to_list()
    assert {'v_near', 'v_far'} <= set(result['variant_labels'][0].to_list())


def test_spark_a_null_transcript_score_ranks_first_and_survives() -> None:
    """`transcriptScore` is null when either factor is null, and spark's ASC NULLS FIRST gives
    those rank 1. polars' bare `rank` would return null and the `rank <= 50` filter would drop
    the variant entirely."""
    variants = pl.LazyFrame(
        [
            _variant(
                variantId='v_null',
                transcriptConsequences=[{'targetId': 'T1', 'consequenceScore': None, 'distanceFromFootprint': 10}],
            )
        ],
        schema=VARIANT_SCHEMA,
    )

    result = variant_labels_by_target(variants).collect()

    assert 'v_null' in result['variant_labels_5'][0].to_list()


def test_spark_a_variant_with_no_consequences_contributes_nothing() -> None:
    """Spark's `explode` drops null and empty arrays; polars' would emit a null row."""
    variants = pl.LazyFrame(
        [
            _variant(variantId='v_empty', transcriptConsequences=[]),
            _variant(variantId='v_none', transcriptConsequences=None),
        ],
        schema=VARIANT_SCHEMA,
    )

    assert variant_labels_by_target(variants).collect().height == 0


def test_target_index_emits_exactly_one_document_per_target() -> None:
    variants = pl.LazyFrame(
        [_variant(variantId='v1'), _variant(variantId='v2')],
        schema=VARIANT_SCHEMA,
    )
    associations = pl.LazyFrame(
        {'associationId': ['D1-T1', 'D2-T1'], 'targetId': ['T1', 'T1'], 'diseaseId': ['D1', 'D2'], 'score': [0.8, 0.2]}
    )
    d_lut = pl.LazyFrame(
        {'diseaseId': ['D1', 'D2'], 'disease_labels': [['asthma'], ['eczema']]},
        schema={'diseaseId': pl.String, 'disease_labels': LIST_STR},
    )

    result = build_target_index(**_inputs(variants=variants, associations=associations, d_lut=d_lut)).collect()

    assert result.height == 1


def test_target_multiplier_is_log1p_of_the_mean_score_plus_one() -> None:
    import math

    result = build_target_index(**_inputs()).collect()

    assert result['multiplier'][0] == math.log1p(0.8) + 1.0


def test_target_with_no_associations_falls_back_to_the_default_multiplier() -> None:
    associations = pl.LazyFrame(
        {'associationId': [], 'targetId': [], 'diseaseId': [], 'score': []},
        schema={'associationId': pl.String, 'targetId': pl.String, 'diseaseId': pl.String, 'score': pl.Float64},
    )

    result = build_target_index(**_inputs(associations=associations)).collect()

    assert result['multiplier'][0] == 0.01


def test_target_with_variants_but_no_associations_gets_no_variant_terms() -> None:
    """46,072 of 78,733 targets on the real release have variants but no association. The
    pyspark original joins variant labels onto the association aggregate BEFORE joining that
    aggregate onto targets, so a target absent from `associations` never receives variant
    labels, even though it has variants of its own -- joining variant labels onto `targets`
    directly would give it terms spark never did."""
    associations = pl.LazyFrame(
        {'associationId': [], 'targetId': [], 'diseaseId': [], 'score': []},
        schema={'associationId': pl.String, 'targetId': pl.String, 'diseaseId': pl.String, 'score': pl.Float64},
    )

    result = build_target_index(**_inputs(associations=associations)).collect()

    assert result['terms'][0].to_list() == []
    assert result['terms25'][0].to_list() == []
    assert result['terms5'][0].to_list() == []
