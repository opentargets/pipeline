from pathlib import Path

import polars as pl

from pts.transformers.clinical_mira.moa import literature
from pts.transformers.clinical_mira.moa.literature import (
    clinical_parents,
    join_genes,
    pass_file,
    sentence_pass,
)

PARENTS = pl.DataFrame({'id': ['CHEMBL1', 'CHEMBL1S'], 'parent': ['CHEMBL1', 'CHEMBL1']})


def match_row(**kw):
    row = {
        'pmid': '1',
        'pmcid': None,
        'section': 'abstract',
        'year': 2020,
        'type': 'CD',
        'mappedId': 'CHEMBL1',
        'label': 'drug',
        'text': 'a sentence about the drug',
        'isValid': True,
    }
    return row | kw


def match_frame(rows):
    return pl.DataFrame(
        rows,
        schema={
            'pmid': pl.String,
            'pmcid': pl.String,
            'section': pl.String,
            'year': pl.Int64,
            'type': pl.String,
            'mappedId': pl.String,
            'label': pl.String,
            'text': pl.String,
            'isValid': pl.Boolean,
        },
    )


def test_pass_file_keeps_valid_cd_of_clinical_parents():
    match = match_frame([
        match_row(text='kept', mappedId='CHEMBL1S'),
        match_row(text='invalid', isValid=False),
        match_row(text='other drug', mappedId='CHEMBL9'),
        match_row(text='a gene', type='GP', mappedId='ENSG1'),
        match_row(text='a disease', type='DS', mappedId='EFO1'),
        match_row(text='pmc only', pmid=None, pmcid='PMC7'),
    ])
    cd, _ = pass_file(match, PARENTS)
    assert sorted(cd['text']) == ['kept', 'pmc only']
    assert set(cd['parent']) == {'CHEMBL1'}
    assert cd.columns == literature.CD_COLUMNS
    assert sorted(cd['pub']) == ['1', 'PMC7']


def test_pass_file_caps_rows_per_parent_preferring_cue_then_section():
    rows = [match_row(text=f'plain {i}', section='title') for i in range(300)]
    rows.append(match_row(text='it inhibits the target', section='results'))
    cd, _ = pass_file(match_frame(rows), PARENTS)
    assert cd.height == 300
    assert cd['text'][0] == 'it inhibits the target'
    assert cd['cue'][0]
    assert not cd['cue'][1]


def test_pass_file_orders_by_section_rank_within_cue():
    rows = [
        match_row(text='in results', section='results'),
        match_row(text='in title', section='title'),
        match_row(text='in unknown', section='zzz'),
        match_row(text='in discuss', section='discuss,x'),
    ]
    cd, _ = pass_file(match_frame(rows), PARENTS)
    assert cd['text'].to_list() == ['in title', 'in discuss', 'in results', 'in unknown']
    assert cd['srank'].to_list() == [0, 4, 5, 9]


def test_pass_file_reduces_gp_to_pub_hash_gene():
    match = match_frame([
        match_row(type='GP', mappedId='ENSG1', text='g'),
        match_row(type='GP', mappedId='ENSG1', text='g', label='again'),
        match_row(type='GP', mappedId='ENSG2', text='g'),
        match_row(type='GP', mappedId='ENSG3', text='g', isValid=False),
    ])
    _, gp = pass_file(match, PARENTS)
    assert gp.columns == ['pub', 'h', 'gene']
    assert sorted(gp['gene']) == ['ENSG1', 'ENSG2']


def test_join_genes_attaches_sorted_unique_genes(tmp_path: Path):
    (tmp_path / 'cd').mkdir()
    (tmp_path / 'gp').mkdir()
    cd, _ = pass_file(match_frame([match_row(text='s1'), match_row(text='s2')]), PARENTS)
    cd.write_parquet(tmp_path / 'cd' / 'a.parquet')
    h1 = cd.filter(pl.col('text') == 's1')['h'][0]
    pl.DataFrame({
        'pub': ['1', '1', '1', '2'],
        'h': [h1, h1, h1, h1],
        'gene': ['ENSG2', 'ENSG1', 'ENSG2', 'ENSG9'],
    }).write_parquet(tmp_path / 'gp' / 'a.parquet')
    out = join_genes(tmp_path / 'cd', tmp_path / 'gp').sort('text')
    assert out['genes'].to_list() == [['ENSG1', 'ENSG2'], None]


def test_clinical_parents_drops_preclinical_and_maps_salts_to_parent():
    molecules = pl.DataFrame({
        'id': ['P1', 'S1', 'P2', 'S2', 'P3', 'P4'],
        'parentId': [None, 'P1', None, 'P2', None, None],
        'maximumClinicalStage': ['APPROVAL', 'PRECLINICAL', 'PRECLINICAL', 'APPROVAL', 'EARLY_PHASE_1', 'UNKNOWN'],
    })
    out = clinical_parents(molecules).sort('id')
    assert out.to_dicts() == [
        {'id': 'P1', 'parent': 'P1'},
        {'id': 'P3', 'parent': 'P3'},
        {'id': 'S1', 'parent': 'P1'},
    ]


def write_match(path: Path, text: str) -> None:
    match_frame([match_row(text=text)]).with_columns(extra=pl.lit(1)).write_parquet(path)


def test_sentence_pass_skips_written_files(tmp_path: Path):
    match_dir = tmp_path / 'match'
    match_dir.mkdir()
    write_match(match_dir / 'part-0001-x.snappy.parquet', 'first')
    write_match(match_dir / 'part-0002-x.snappy.parquet', 'second')
    scratch = tmp_path / 'scratch'

    cd_dir, gp_dir = sentence_pass(str(match_dir), PARENTS, 2, scratch, None)
    assert sorted(p.name for p in cd_dir.glob('*.parquet')) == [
        'part-0001-x.snappy.parquet',
        'part-0002-x.snappy.parquet',
    ]
    assert len(list(gp_dir.glob('*.parquet'))) == 2
    assert not list(cd_dir.glob('*.tmp'))

    marker = cd_dir / 'part-0001-x.snappy.parquet'
    marker.write_bytes(b'sentinel')
    write_match(match_dir / 'part-0001-x.snappy.parquet', 'changed')
    sentence_pass(str(match_dir), PARENTS, 2, scratch, None)
    assert marker.read_bytes() == b'sentinel'
