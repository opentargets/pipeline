"""Tests for the acronym rule and the spelling retry in `add_efo_mapping`."""

from pathlib import Path

import polars as pl
import pytest
from loguru import logger

from pts.transformers.evidence.utils.ontology import MAPPED_ID_COLUMN, add_efo_mapping
from pts.transformers.utils.disease_mapping.normalise import normalise_symbol, normalise_term
from tests.evidence.fixtures import write_ontoma_lut


def _label_lut(tmp_path: Path, label: str, entity_id: str, is_acronym: bool) -> str:
    return write_ontoma_lut(tmp_path / 'label_lut', 'label', normalise_symbol(label), entity_id, is_acronym)


def test_acronym_label_not_mapped(tmp_path: Path) -> None:
    lut = _label_lut(tmp_path, 'ALS', 'MONDO_0004976', True)
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS'], 'diseaseFromSourceId': ['x']})
    out = add_efo_mapping(evidence, lut, id_col_name=None).collect()
    assert out.rows() == [('ALS', 'x', None)]


def test_unflagged_short_label_mapped(tmp_path: Path) -> None:
    lut = _label_lut(tmp_path, 'HIV', 'MONDO_0005109', False)
    evidence = pl.LazyFrame({'diseaseFromSource': ['HIV'], 'diseaseFromSourceId': ['x']})
    out = add_efo_mapping(evidence, lut, id_col_name=None).collect()
    assert out.rows() == [('HIV', 'x', 'MONDO_0005109')]


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


def test_keep_acronyms_maps_flagged_label(tmp_path: Path) -> None:
    lut = _label_lut(tmp_path, 'ALS', 'MONDO_0004976', True)
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS', 'unknown'], 'diseaseFromSourceId': ['x', 'y']})
    out = add_efo_mapping(evidence, lut, id_col_name=None, keep_acronyms=True).collect()
    assert out.rows() == [('ALS', 'x', 'MONDO_0004976'), ('unknown', 'y', None)]


def _term_lut(tmp_path: Path, label: str, entity_id: str, is_acronym: bool = False) -> str:
    return write_ontoma_lut(tmp_path / 'label_lut', 'label', normalise_term(label), entity_id, is_acronym)


@pytest.mark.parametrize(('spelling_variants', 'expected'), [(False, None), (True, 'EFO_0000400')])
def test_unmapped_label_retried_in_american_spelling(tmp_path: Path, spelling_variants: bool, expected: str) -> None:
    lut = _term_lut(tmp_path, 'diabetic macular edema', 'EFO_0000400')
    evidence = pl.LazyFrame({'diseaseFromSource': ['Diabetic macular oedema (DME)'], 'diseaseFromSourceId': ['x']})
    out = add_efo_mapping(evidence, lut, id_col_name=None, spelling_variants=spelling_variants).collect()
    assert out[MAPPED_ID_COLUMN].to_list() == [expected]


def test_label_not_retried_when_id_maps(tmp_path: Path) -> None:
    label_lut = _term_lut(tmp_path, 'noonan syndrome', 'MONDO_BROAD')
    id_lut = write_ontoma_lut(tmp_path / 'id_lut', 'id', normalise_symbol('OMIM:616564'), 'MONDO_PRECISE')
    evidence = pl.LazyFrame({'diseaseFromSource': ['Noonan syndrome (AD)'], 'diseaseFromSourceId': ['OMIM:616564']})
    out = add_efo_mapping(evidence, label_lut, disease_id_lut_path=id_lut, spelling_variants=True).collect()
    assert out[MAPPED_ID_COLUMN].to_list() == ['MONDO_PRECISE']


@pytest.mark.parametrize(('keep_acronyms', 'expected'), [(False, None), (True, 'MONDO_0004976')])
def test_retried_acronym_follows_keep_acronyms(tmp_path: Path, keep_acronyms: bool, expected: str) -> None:
    lut = _term_lut(tmp_path, 'ALS', 'MONDO_0004976', is_acronym=True)
    evidence = pl.LazyFrame({'diseaseFromSource': ['ALS (AD)'], 'diseaseFromSourceId': ['x']})
    out = add_efo_mapping(
        evidence, lut, id_col_name=None, keep_acronyms=keep_acronyms, spelling_variants=True
    ).collect()
    assert out[MAPPED_ID_COLUMN].to_list() == [expected]
