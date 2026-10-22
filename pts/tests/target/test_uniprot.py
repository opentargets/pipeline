"""The UniProt flat file parser and builder."""

from __future__ import annotations

import gzip
from pathlib import Path

import polars as pl

from pts.schemas.target import LOCATION, UNIPROT_ENTRY
from pts.transformers.target.hpa import build_hpa
from pts.transformers.target.uniprot import _parse_record, build_uniprot, read_uniprot, read_uniprot_ssl
from tests.target.rows import rows

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

SECOND_ENTRY = """\
ID   OTHER_HUMAN          Reviewed;          50 AA.
AC   O00001;
DE   RecName: Full=Other protein;
DE            Short=OP;
GN   Name=OTHER1;
CC   -!- SUBCELLULAR LOCATION: [Isoform 2]: Cell membrane; Multi-pass membrane
CC       protein. Golgi apparatus, trans-Golgi network membrane. (Microbial
CC       infection) Host, Nucleus.\
"""

SSL_LINES = [
    'Subcellular location ID\tName\tCategory',
    'SL-0191\tNucleus\tCellular component',
    'SL-0086\tCytoplasm\tCellular component',
    'SL-0039\tCell membrane\tCellular component',
    'SL-0268\ttrans-Golgi network membrane\tCellular component',
]


def _lines(entry: str) -> list[str]:
    return entry.splitlines()


def _write(tmp_path: Path) -> tuple[str, str]:
    flat = tmp_path / 'uniprot.txt.gz'
    flat.write_bytes(gzip.compress(f'{MINIMAL_ENTRY}\n//\n{SECOND_ENTRY}\n//\n'.encode()))
    ssl = tmp_path / 'uniprot-ssl.tsv.gz'
    ssl.write_bytes(gzip.compress(('\n'.join(SSL_LINES) + '\n').encode()))
    return str(flat), str(ssl)


def _built(tmp_path: Path) -> dict:
    flat, ssl = _write(tmp_path)
    return {r['uniprotId']: r for r in build_uniprot(read_uniprot(flat), read_uniprot_ssl(ssl)).to_dicts()}


def test_parse_record_id():
    assert _parse_record(_lines(MINIMAL_ENTRY))['id'] == 'PROT_HUMAN'


def test_parse_record_reviewed():
    assert _parse_record(_lines(MINIMAL_ENTRY))['reviewed'] is True
    assert _parse_record(_lines(MINIMAL_ENTRY.replace('Reviewed;', 'Unreviewed;')))['reviewed'] is False


def test_parse_record_accessions():
    assert _parse_record(_lines(MINIMAL_ENTRY))['accessions'] == ['P12345', 'Q99999']


def test_parse_record_names():
    assert 'Test protein' in _parse_record(_lines(MINIMAL_ENTRY))['names']


def test_parse_record_synonyms():
    assert 'Alt protein' in _parse_record(_lines(MINIMAL_ENTRY))['synonyms']


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
    assert any('Catalyzes the reaction' in f for f in _parse_record(_lines(MINIMAL_ENTRY))['functions'])


def test_parse_record_locations():
    locations = [loc['location'] for loc in _parse_record(_lines(MINIMAL_ENTRY))['locations']]
    assert 'Nucleus' in locations
    assert 'Cytoplasm' in locations


def test_parse_record_location_modifier():
    entry = """\
ID   PROT_HUMAN           Reviewed;         100 AA.
AC   P12345;
CC   -!- SUBCELLULAR LOCATION: [Isoform 4]: Cytoplasm. Nucleus.\
"""
    by_location = {loc['location']: loc['targetModifier'] for loc in _parse_record(_lines(entry))['locations']}
    assert by_location['Cytoplasm'] == 'Isoform 4'
    assert by_location['Nucleus'] is None


def test_read_uniprot_one_row_per_entry(tmp_path: Path) -> None:
    flat, _ = _write(tmp_path)
    entries = read_uniprot(flat)
    assert entries.schema == UNIPROT_ENTRY
    assert entries['id'].to_list() == ['PROT_HUMAN', 'OTHER_HUMAN']


def test_entry_keyed_by_primary_accession_with_every_accession_as_protein_id(tmp_path: Path) -> None:
    entry = _built(tmp_path)['P12345']
    assert entry['proteinIds'] == [
        {'id': 'P12345', 'source': 'uniprot_obsolete'},
        {'id': 'Q99999', 'source': 'uniprot_obsolete'},
    ]


def test_synonym_lists(tmp_path: Path) -> None:
    entry = _built(tmp_path)['O00001']
    assert [s['label'] for s in entry['nameSynonyms']] == ['Other protein']
    assert [s['label'] for s in entry['symbolSynonyms']] == ['OP', 'OTHER1']
    assert [s['label'] for s in entry['synonyms']] == ['Other protein', 'OP', 'OTHER1']
    assert {s['source'] for s in entry['synonyms']} == {'uniprot'}


def test_db_xrefs_split_into_id_and_source(tmp_path: Path) -> None:
    assert _built(tmp_path)['P12345']['dbXrefs'] == [
        {'id': 'CHEMBL123', 'source': 'ChEMBL'},
        {'id': 'ENST00001', 'source': 'Ensembl'},
        {'id': '1ABC', 'source': 'PDB'},
    ]


def test_locations_map_by_leading_words_then_after_the_last_comma(tmp_path: Path) -> None:
    assert _built(tmp_path)['O00001']['subcellularLocations'] == [
        {
            'location': 'Cell membrane; Multi-pass membrane protein',
            'source': 'uniprot',
            'termSL': 'SL-0039',
            'labelSL': 'Cellular component',
            'targetModifier': 'Isoform 2',
        },
        {
            'location': 'Golgi apparatus, trans-Golgi network membrane',
            'source': 'uniprot',
            'termSL': None,
            'labelSL': None,
            'targetModifier': None,
        },
        {
            'location': '(Microbial infection) Host, Nucleus',
            'source': 'uniprot',
            'termSL': 'SL-0191',
            'labelSL': 'Cellular component',
            'targetModifier': None,
        },
    ]


def test_location_struct_matches_hpa(tmp_path: Path) -> None:
    flat, ssl = _write(tmp_path)
    uniprot = build_uniprot(read_uniprot(flat), read_uniprot_ssl(ssl))
    hpa = build_hpa(
        pl.DataFrame(
            {
                'Gene': ['ENSG1'],
                'Main location': ['Cytosol'],
                'Additional location': [None],
                'Extracellular location': [None],
            },
            schema=dict.fromkeys(['Gene', 'Main location', 'Additional location', 'Extracellular location'], pl.String),
        ),
        pl.DataFrame({'HPA_location': ['Cytosol'], 'termSL': ['SL-0091'], 'labelSL': ['Cytosol']}),
    )
    assert uniprot.schema['subcellularLocations'] == hpa.schema['locations'] == pl.List(LOCATION)


def test_matches_the_spark_output(tmp_path: Path) -> None:
    flat, ssl = _write(tmp_path)
    built = build_uniprot(read_uniprot(flat), read_uniprot_ssl(ssl))
    assert rows(built, 'uniprotId', ('subcellularLocations',)) == SPARK_ROWS


#: The spark output for this fixture, as `rows` normalises it.
SPARK_ROWS = [
    {
        'uniprotId': 'O00001',
        'functionDescriptions': [],
        'dbXrefs': [],
        'subcellularLocations': [
            {
                'location': '(Microbial infection) Host, Nucleus',
                'source': 'uniprot',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            },
            {
                'location': 'Cell membrane; Multi-pass membrane protein',
                'source': 'uniprot',
                'termSL': 'SL-0039',
                'labelSL': 'Cellular component',
                'targetModifier': 'Isoform 2',
            },
            {
                'location': 'Golgi apparatus, trans-Golgi network membrane',
                'source': 'uniprot',
                'termSL': None,
                'labelSL': None,
                'targetModifier': None,
            },
        ],
        'proteinIds': [{'id': 'O00001', 'source': 'uniprot_obsolete'}],
        'synonyms': [
            {'label': 'Other protein', 'source': 'uniprot'},
            {'label': 'OP', 'source': 'uniprot'},
            {'label': 'OTHER1', 'source': 'uniprot'},
        ],
        'symbolSynonyms': [{'label': 'OP', 'source': 'uniprot'}, {'label': 'OTHER1', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'Other protein', 'source': 'uniprot'}],
    },
    {
        'uniprotId': 'P12345',
        'functionDescriptions': ['Catalyzes the reaction.'],
        'dbXrefs': [
            {'id': 'CHEMBL123', 'source': 'ChEMBL'},
            {'id': 'ENST00001', 'source': 'Ensembl'},
            {'id': '1ABC', 'source': 'PDB'},
        ],
        'subcellularLocations': [
            {
                'location': 'Cytoplasm',
                'source': 'uniprot',
                'termSL': 'SL-0086',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            },
            {
                'location': 'Nucleus',
                'source': 'uniprot',
                'termSL': 'SL-0191',
                'labelSL': 'Cellular component',
                'targetModifier': None,
            },
        ],
        'proteinIds': [{'id': 'P12345', 'source': 'uniprot_obsolete'}, {'id': 'Q99999', 'source': 'uniprot_obsolete'}],
        'synonyms': [
            {'label': 'Test protein', 'source': 'uniprot'},
            {'label': 'GENE1', 'source': 'uniprot'},
            {'label': 'SYN1', 'source': 'uniprot'},
        ],
        'symbolSynonyms': [{'label': 'GENE1', 'source': 'uniprot'}, {'label': 'SYN1', 'source': 'uniprot'}],
        'nameSynonyms': [{'label': 'Test protein', 'source': 'uniprot'}, {'label': 'Alt protein', 'source': 'uniprot'}],
    },
]
