"""Tests for the uniprot transformer."""

from __future__ import annotations

from pts.transformers.uniprot import _parse_record

MINIMAL_ENTRY = """\
ID   PROT_HUMAN           Reviewed;         100 AA.
AC   P12345; Q99999;
DE   RecName: Full=Test protein;
DE   AltName: Full=Alt protein;
GN   Name=GENE1; Synonyms=SYN1;
DR   ChEMBL; CHEMBL123; -.
DR   Ensembl; ENST00001; ENSP00001; ENSG00001.
DR   PDB; 1ABC; X-ray; -.
CC   -!- FUNCTION: Catalyzes the reaction.
CC   -!- SUBCELLULAR LOCATION: Nucleus. Cytoplasm.\
"""


def _lines(entry: str) -> list[str]:
    return entry.splitlines()


def test_parse_record_id():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert rec['id'] == 'PROT_HUMAN'


def test_parse_record_accessions():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert rec['accessions'] == ['P12345', 'Q99999']


def test_parse_record_names():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert 'Test protein' in rec['names']


def test_parse_record_synonyms():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert 'Alt protein' in rec['synonyms']


def test_parse_record_symbol_synonyms():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert 'GENE1' in rec['symbolSynonyms']
    assert 'SYN1' in rec['symbolSynonyms']


def test_parse_record_db_xrefs():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert 'CHEMBL123 ChEMBL' in rec['dbXrefs']
    assert 'ENST00001 Ensembl' in rec['dbXrefs']
    assert '1ABC PDB' in rec['dbXrefs']


def test_parse_record_functions():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    assert any('Catalyzes the reaction' in f for f in rec['functions'])


def test_parse_record_locations():
    rec = _parse_record(_lines(MINIMAL_ENTRY))
    locations = [loc['location'] for loc in rec['locations']]
    assert 'Nucleus' in locations
    assert 'Cytoplasm' in locations


def test_parse_record_location_modifier():
    entry = """\
ID   PROT_HUMAN           Reviewed;         100 AA.
AC   P12345;
CC   -!- SUBCELLULAR LOCATION: [Isoform 4]: Cytoplasm. Nucleus.\
"""
    rec = _parse_record(_lines(entry))
    by_location = {loc['location']: loc['targetModifier'] for loc in rec['locations']}
    assert by_location['Cytoplasm'] == 'Isoform 4'
    assert by_location['Nucleus'] is None
