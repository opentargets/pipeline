"""Tests for COSMIC Cancer Gene Census evidence."""

from __future__ import annotations

import polars as pl

from pts.transformers.evidence.cosmic import CANCER_GENE_CENSUS, DIRECTION_ON_TARGET, DIRECTION_ON_TRAIT


def test_score_is_the_resource_score() -> None:
    df = pl.DataFrame({'resourceScore': [0.5, None]})
    assert df.select(CANCER_GENE_CENSUS.score)['resourceScore'].to_list() == [0.5, None]


def test_direction_on_trait_is_risk_when_disease_present() -> None:
    df = pl.DataFrame({'diseaseId': ['MONDO:1', None]})
    assert df.select(DIRECTION_ON_TRAIT.alias('d'))['d'].to_list() == ['risk', None]


def test_cancer_gene_census_direction_on_target_lookup() -> None:
    df = pl.DataFrame({'TSorOncogene': ['oncogene', 'tsg', 'bivalent', None]})
    assert df.select(DIRECTION_ON_TARGET.alias('d'))['d'].to_list() == ['GoF', 'LoF', None, None]


