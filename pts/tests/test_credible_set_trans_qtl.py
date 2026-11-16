"""The TRANS_QTL rule: QTL credible sets whose every variant is trans to the gene of their study."""

from typing import Any

import polars as pl
import pytest

from pts.transformers.credible_set.flags import CredibleSetFlag
from pts.transformers.credible_set.trans_qtl import trans_qtls

TSS = 10_000_000
"""The TSS of gene `g`, on chromosome 1."""
TARGETS = pl.LazyFrame({
    'id': ['g'],
    'tss': [TSS],
    'genomicLocation': [{'chromosome': '1', 'start': TSS, 'end': TSS + 1, 'strand': 1}],
})


def _sets(*rows: dict[str, Any]) -> pl.LazyFrame:
    defaults = {'qualityControls': [CredibleSetFlag.TOP_HIT.value]}
    return pl.LazyFrame(
        [{**defaults, **row} for row in rows],
        schema={
            'id': pl.String,
            'studyId': pl.String,
            'leadVariantId': pl.String,
            'qualityControls': pl.List(pl.String),
        },
    )


def _ids(frame: pl.LazyFrame) -> set[str]:
    ids = frame.collect()
    assert ids.columns == ['id']
    return set(ids['id'])


def _qtl_studies(**rows: tuple[str, str | None]) -> pl.LazyFrame:
    """Studies of `studyId=(studyType, geneId)` keyword pairs."""
    return pl.LazyFrame(
        [{'studyId': study, 'studyType': kind, 'geneId': gene} for study, (kind, gene) in rows.items()],
        schema={'studyId': pl.String, 'studyType': pl.String, 'geneId': pl.String},
    )


def _trans(
    variant_ids: list[str], study_type: str = 'pqtl', gene: str | None = 'g', threshold: int = 5_000_000
) -> bool:
    sets = _sets({'id': 'cs', 'studyId': 's', 'leadVariantId': variant_ids[0]})
    variants = pl.LazyFrame({'credibleSetId': ['cs'] * len(variant_ids), 'variantId': variant_ids})
    return _ids(trans_qtls(sets, variants, _qtl_studies(s=(study_type, gene)), TARGETS, threshold)) == {'cs'}


FAR = [f'1_{TSS + 5_000_001}_A_T', f'1_{TSS - 6_000_000}_C_G']


def test_a_qtl_with_every_variant_beyond_the_threshold_is_trans() -> None:
    assert _trans(FAR)


def test_a_qtl_with_one_variant_within_the_threshold_is_not_trans() -> None:
    assert not _trans([*FAR, f'1_{TSS + 5_000_000}_A_T'])


def test_a_qtl_on_another_chromosome_than_its_gene_is_trans() -> None:
    assert _trans([f'2_{TSS}_A_T'])


@pytest.mark.parametrize(
    ('study_type', 'gene'),
    [('gwas', 'g'), ('pqtl', None), ('pqtl', 'unknown')],
    ids=['gwas', 'no gene', 'unknown gene'],
)
def test_a_gwas_or_a_qtl_without_a_located_gene_is_never_trans(study_type: str, gene: str | None) -> None:
    assert not _trans([f'2_{TSS}_A_T'], study_type, gene)


def test_the_trans_threshold_is_a_setting() -> None:
    assert _trans([f'1_{TSS + 1_001}_A_T'], threshold=1_000)
