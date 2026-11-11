"""Tests for the acronym rule in `add_efo_mapping`."""

from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from pts.transformers.evidence.utils.ontology import add_efo_mapping
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol
from tests.evidence.fixtures import write_ontoma_lut


def _label_lut(tmp_path: Path, label: str, entity_id: str, is_acronym: bool) -> str:
    return write_ontoma_lut(tmp_path / 'label_lut', 'label', normalise_symbol(label), entity_id, is_acronym)


def test_acronym_label_not_mapped(tmp_path: Path) -> None:
    lut = _label_lut(tmp_path, 'ALS', 'MONDO_0004976', True)
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS'], 'diseaseFromSourceId': ['x']})
    out = add_efo_mapping(evidence, lut, id_col_name=None).collect()
    assert out.rows() == [('ALS', 'x', None)]


def test_unflagged_short_label_mapped(tmp_path: Path) -> None:
    lut = _label_lut(tmp_path, 'HIV', 'EFO_0000764', False)
    evidence = pl.LazyFrame({'diseaseFromSource': ['HIV'], 'diseaseFromSourceId': ['x']})
    out = add_efo_mapping(evidence, lut, id_col_name=None).collect()
    assert out.rows() == [('HIV', 'x', 'EFO_0000764')]


def test_id_mapping_unaffected(tmp_path: Path) -> None:
    label_lut = _label_lut(tmp_path, 'ALS', 'MONDO_0004976', True)
    id_lut = write_ontoma_lut(tmp_path / 'id_lut', 'id', normalise_symbol('Orphanet:803'), 'MONDO_0004976')
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS'], 'diseaseFromSourceId': ['Orphanet:803']})
    out = add_efo_mapping(evidence, label_lut, disease_id_lut_path=id_lut).collect()
    assert out.rows() == [('ALS', 'Orphanet:803', 'MONDO_0004976')]


@pytest.mark.parametrize(('is_acronym', 'expected'), [(True, 1), (False, 0)])
def test_logs_dropped_acronym_ids(tmp_path: Path, is_acronym: bool, expected: int) -> None:
    lut = _label_lut(tmp_path, 'ALS', 'MONDO_0004976', is_acronym)
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS', 'unknown'], 'diseaseFromSourceId': ['x', 'y']})
    messages: list[str] = []
    sink = logger.add(lambda m: messages.append(str(m)), level='INFO')
    try:
        add_efo_mapping(evidence, lut, id_col_name=None).collect()
    finally:
        logger.remove(sink)
    assert any(f'{expected} evidence rows lost' in m for m in messages)
