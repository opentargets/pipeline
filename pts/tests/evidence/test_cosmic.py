"""Tests for COSMIC Cancer Gene Census evidence."""

from __future__ import annotations

import gzip
import json
from pathlib import Path

import polars as pl

from pts.transformers.evidence.cosmic import CANCER_GENE_CENSUS, DIRECTION_ON_TARGET, DIRECTION_ON_TRAIT, cosmic
from tests.evidence.fixtures import write_disease, write_literature, write_target


def test_score_is_the_resource_score() -> None:
    df = pl.DataFrame({'resourceScore': [0.5, None]})
    assert df.select(CANCER_GENE_CENSUS.score)['resourceScore'].to_list() == [0.5, None]


def test_direction_on_trait_is_risk_when_disease_present() -> None:
    df = pl.DataFrame({'diseaseId': ['MONDO:1', None]})
    assert df.select(DIRECTION_ON_TRAIT.alias('d'))['d'].to_list() == ['risk', None]


def test_cancer_gene_census_direction_on_target_lookup() -> None:
    df = pl.DataFrame({'TSorOncogene': ['oncogene', 'tsg', 'bivalent', None]})
    assert df.select(DIRECTION_ON_TARGET.alias('d'))['d'].to_list() == ['GoF', 'LoF', None, None]


def test_cosmic_end_to_end(tmp_path: Path) -> None:
    path = tmp_path / 'src' / 'cosmic.json.gz'
    path.parent.mkdir(parents=True)
    row = {
        'datasourceId': 'cancer_gene_census',
        'targetFromSourceId': 'A1',
        'diseaseFromSourceMappedId': 'EFO_1',
        'resourceScore': 1.0,
    }
    with gzip.open(path, 'wt') as f:
        f.write(json.dumps(row) + '\n')
    source = {
        'evidence': str(path),
        'disease': write_disease(tmp_path / 'src' / 'disease', ['EFO_1']),
        'target': write_target(
            tmp_path / 'src' / 'target', ['ENSG1'], {'ENSG1': 'A1'}, hallmarks={'ENSG1': ['Is an oncogene']}
        ),
        'literature': write_literature(tmp_path / 'src' / 'literature', {}),
    }
    destination = {'evidence': str(tmp_path / 'out' / 'valid'), 'failed_evidence': str(tmp_path / 'out' / 'failed')}

    cosmic(source, destination, {}, None)

    valid = pl.read_parquet(destination['evidence'])
    assert valid.select('targetId', 'score', 'directionOnTrait', 'directionOnTarget').rows() == [
        ('ENSG1', 1.0, 'risk', 'GoF')
    ]
