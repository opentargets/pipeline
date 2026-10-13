"""Tests for UniProt variants evidence."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

from pts.transformers.evidence.uniprot_variants import UNIPROT_VARIANTS, uniprot_variants
from tests.evidence.fixtures import write_disease, write_literature, write_target


def test_score_is_the_confidence_score() -> None:
    df = pl.DataFrame({'confidence': ['high', 'medium', 'low', None]})
    assert df.select(UNIPROT_VARIANTS.score.alias('s'))['s'].to_list() == [1.0, 0.5, None, None]


def test_long_variant_ids_are_hashed(tmp_path: Path) -> None:
    src = tmp_path / 'src'
    src.mkdir()
    long_id = '1_100_' + 'A' * 400 + '_C'
    rows = [
        {
            'datasourceId': 'uniprot_variants',
            'targetFromSourceId': 'A1',
            'diseaseFromSourceMappedId': 'EFO_1',
            'diseaseFromSource': 'd',
            'variantRsId': 'rs1',
            'variantId': variant_id,
            'confidence': 'high',
            'literature': ['1'],
        }
        for variant_id in ('1_100_A_C', long_id)
    ]
    (src / 'uniprot_variants.json').write_text('\n'.join(json.dumps(r) for r in rows) + '\n')
    dst = tmp_path / 'out'

    uniprot_variants(
        {
            'evidence': str(src / 'uniprot_variants.json'),
            'disease': write_disease(src / 'disease', ['EFO_1']),
            'target': write_target(src / 'target', ['ENSG1'], {'ENSG1': 'A1'}),
            'literature': write_literature(src / 'literature', {'1': '2020-01-01'}),
        },
        {'evidence': str(dst / 'valid'), 'failed_evidence': str(dst / 'failed')},
        {},
        None,
    )

    valid = pl.read_parquet(dst / 'valid').sort('variantId')
    assert valid['score'].to_list() == [1.0, 1.0]
    short, hashed = valid['variantId'].to_list()
    assert short == '1_100_A_C'
    assert hashed.startswith('OTVAR_')
    assert len(hashed) < 100
