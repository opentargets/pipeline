import hashlib
import json
import random
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.clinical_mira.moa import literature
from pts.transformers.clinical_mira.moa.literature import (
    clinical_parents,
    content_key,
    join_genes,
    pass_file,
    read_parquet_dir,
    requests,
    select_sentences,
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


def test_join_genes_without_any_gene_mention_leaves_genes_null(tmp_path: Path):
    (tmp_path / 'cd').mkdir()
    (tmp_path / 'gp').mkdir()
    cd, gp = pass_file(match_frame([match_row(text='s1'), match_row(text='s2')]), PARENTS)
    cd.write_parquet(tmp_path / 'cd' / 'a.parquet')
    cd.write_parquet(tmp_path / 'cd' / 'b.parquet')
    gp.write_parquet(tmp_path / 'gp' / 'a.parquet')
    gp.write_parquet(tmp_path / 'gp' / 'b.parquet')
    assert gp.height == 0

    out = join_genes(tmp_path / 'cd', tmp_path / 'gp')

    assert out.height == 4
    assert out['genes'].dtype == pl.List(pl.String)
    assert out['genes'].to_list() == [None] * 4


def test_read_parquet_dir_concatenates_the_columns_of_every_file(tmp_path: Path):
    pl.DataFrame({'id': ['a'], 'name': ['x'], 'extra': [1]}).write_parquet(tmp_path / 'part-1.parquet')
    pl.DataFrame({'id': ['b'], 'name': ['y'], 'extra': [2]}).write_parquet(tmp_path / 'part-0.parquet')
    (tmp_path / '_SUCCESS').write_text('')

    out = read_parquet_dir(str(tmp_path), ['id', 'name'])

    assert out.to_dicts() == [{'id': 'b', 'name': 'y'}, {'id': 'a', 'name': 'x'}]


def test_read_parquet_dir_rejects_a_directory_without_parquet_files(tmp_path: Path):
    with pytest.raises(ValueError, match='no parquet files found under'):
        read_parquet_dir(str(tmp_path / 'missing'), ['id'])


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

    cd_dir, gp_dir = sentence_pass(str(match_dir), PARENTS, 2, scratch)
    assert sorted(p.name for p in cd_dir.glob('*.parquet')) == [
        'part-0001-x.snappy.parquet',
        'part-0002-x.snappy.parquet',
    ]
    assert len(list(gp_dir.glob('*.parquet'))) == 2
    assert not list(cd_dir.glob('*.tmp'))

    marker = cd_dir / 'part-0001-x.snappy.parquet'
    marker.write_bytes(b'sentinel')
    write_match(match_dir / 'part-0001-x.snappy.parquet', 'changed')
    sentence_pass(str(match_dir), PARENTS, 2, scratch)
    assert marker.read_bytes() == b'sentinel'


def test_sentence_pass_rejects_a_match_uri_without_parquet_files(tmp_path: Path):
    with pytest.raises(ValueError, match='no parquet files found under'):
        sentence_pass(str(tmp_path / 'missing'), PARENTS, 1, tmp_path / 'scratch')


def _record_reads(monkeypatch: pytest.MonkeyPatch) -> list[tuple[object, list[str]]]:
    reads: list[tuple[object, list[str]]] = []
    real = pl.read_parquet

    def read_parquet(source, *args, columns=None, **kwargs):
        reads.append((source, columns))
        return real(source, *args, columns=columns, **kwargs)

    monkeypatch.setattr(literature.pl, 'read_parquet', read_parquet)
    return reads


def test_sentence_pass_hands_polars_the_uri_of_each_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    match_dir = tmp_path / 'match'
    match_dir.mkdir()
    write_match(match_dir / 'part-0001.parquet', 'first')
    write_match(match_dir / 'part-0002.parquet', 'second')
    reads = _record_reads(monkeypatch)

    sentence_pass(str(match_dir), PARENTS, 1, tmp_path / 'scratch')

    assert sorted(reads) == [
        (str(match_dir / 'part-0001.parquet'), literature.MATCH_COLUMNS),
        (str(match_dir / 'part-0002.parquet'), literature.MATCH_COLUMNS),
    ]


def test_read_parquet_dir_hands_polars_the_uri_of_each_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    pl.DataFrame({'id': ['a'], 'name': ['x']}).write_parquet(tmp_path / 'part-0.parquet')
    reads = _record_reads(monkeypatch)

    read_parquet_dir(str(tmp_path), ['id'])

    assert reads == [(str(tmp_path / 'part-0.parquet'), ['id'])]


SENTENCE_SCHEMA = {
    'parent': pl.String,
    'mappedId': pl.String,
    'pub': pl.String,
    'pmid': pl.String,
    'pmcid': pl.String,
    'section': pl.String,
    'year': pl.Int64,
    'label': pl.String,
    'text': pl.String,
    'h': pl.UInt64,
    'cue': pl.Boolean,
    'srank': pl.Int8,
    'genes': pl.List(pl.String),
}


def sentence_row(**kw):
    row = {
        'parent': 'CHEMBL1',
        'mappedId': 'CHEMBL1',
        'pub': '1',
        'pmid': '1',
        'pmcid': None,
        'section': 'abstract',
        'year': 2020,
        'label': 'Drug',
        'text': 'a sentence about the drug that is long enough',
        'h': 1,
        'cue': False,
        'srank': 1,
        'genes': None,
    }
    return row | kw


def sentence_frame(rows):
    return pl.DataFrame(rows, schema=SENTENCE_SCHEMA)


NAMES = pl.DataFrame({'parent': ['CHEMBL1', 'CHEMBL2'], 'name': ['Aspirin', 'Other']})


def numbered(n, **kw):
    return [
        sentence_row(text=f'sentence number {i:03d} about the drug here', h=i, pub=str(i), pmid=str(i), **kw)
        for i in range(n)
    ]


def tied_rows():
    return [
        sentence_row(text='tied duplicate text one about the drug', pub='4', pmid='4', h=4, mappedId='CHEMBL1S'),
        sentence_row(text='tied duplicate text one about the drug', pub='4', pmid='4', h=4, mappedId='CHEMBL1'),
        sentence_row(text='tied duplicate text two about the drug', pub='5', pmid='5', h=5, section='results'),
        sentence_row(text='tied duplicate text two about the drug', pub='5', pmid='5', h=5, section='abstract'),
        sentence_row(text='tied duplicate text three about the drug', pub='6', pmid='6', h=6, pmcid='PMC2'),
        sentence_row(text='tied duplicate text three about the drug', pub='6', pmid='6', h=6, pmcid='PMC1'),
        sentence_row(text='tied ranking text beta about the drug', pub='8', pmid='8', h=8),
        sentence_row(text='tied ranking text alpha about the drug', pub='8', pmid='8', h=8),
        sentence_row(text='same text twice but different label here', label='B', h=1),
        sentence_row(text='same text twice but different label here', label='A', h=1),
        sentence_row(text='gene sentence one in publication nine', pub='9', pmid='9', h=3, genes=['G1']),
        sentence_row(text='cue sentence one in publication nine', pub='9', pmid='9', h=4, cue=True),
        sentence_row(text='second drug sentence is long enough', parent='CHEMBL2', h=7),
        *numbered(10),
    ]


def test_select_is_deterministic_under_shuffle():
    rows = tied_rows()
    expected = select_sentences(sentence_frame(rows))
    expected_requests = requests(expected, NAMES)
    survivors = expected.filter(pl.col('text').str.contains('tied duplicate')).sort('text')
    assert survivors['mappedId'].to_list() == ['CHEMBL1', 'CHEMBL1', 'CHEMBL1']
    assert survivors['section'].to_list() == ['abstract', 'abstract', 'abstract']
    assert survivors['pmcid'].to_list() == [None, 'PMC1', None]
    rng = random.Random(3)
    for _ in range(20):
        shuffled = rows[:]
        rng.shuffle(shuffled)
        got = select_sentences(sentence_frame(shuffled))
        assert got.equals(expected)
        assert requests(got, NAMES).equals(expected_requests)


def test_select_orders_equal_ranking_keys_by_text():
    out = select_sentences(sentence_frame(tied_rows())).filter(pl.col('text').str.contains('tied ranking'))
    assert out['text'].to_list() == [
        'tied ranking text alpha about the drug',
        'tied ranking text beta about the drug',
    ]


def test_select_tiers_cue_gene_first():
    rows = [
        sentence_row(text='neither cue nor gene sentence here ok', pub='1', pmid='1'),
        sentence_row(text='gene only sentence is long enough ok', pub='2', pmid='2', genes=['G']),
        sentence_row(text='cue only sentence is long enough ok', pub='3', pmid='3', cue=True),
        sentence_row(text='cue and gene sentence is long enough', pub='4', pmid='4', cue=True, genes=['G']),
    ]
    out = select_sentences(sentence_frame(rows))
    assert out['tier'].to_list() == [0, 1, 2, 3]
    assert out['text'].to_list()[0] == 'cue and gene sentence is long enough'


def test_select_spreads_across_publications():
    rows = [sentence_row(text=f'sentence a{i} of the first publication ok', h=i, srank=1) for i in range(3)] + [
        sentence_row(text='sentence of the second publication ok', pub='2', pmid='2', h=9, srank=5)
    ]
    out = select_sentences(sentence_frame(rows))
    assert out['pub'].to_list()[:2] == ['1', '2']
    assert out['nth_in_pub'].to_list() == [0, 0, 1, 2]


def test_select_length_bounds():
    rows = [
        sentence_row(text='x' * 29, h=1),
        sentence_row(text='y' * 30, h=2),
        sentence_row(text='z' * 1200, h=3),
        sentence_row(text='w' * 1201, h=4),
        sentence_row(text='  ' + 'v' * 28 + '  ', h=5),
    ]
    out = select_sentences(sentence_frame(rows))
    assert sorted(out['text'].to_list()) == sorted(['y' * 30, 'z' * 1200])


def test_select_takes_25_per_drug():
    rows = numbered(40) + [
        sentence_row(parent='CHEMBL2', text=f'other drug sentence {i} long enough', h=i) for i in range(3)
    ]
    out = select_sentences(sentence_frame(rows))
    assert out.filter(pl.col('parent') == 'CHEMBL1').height == 25
    assert out.filter(pl.col('parent') == 'CHEMBL2').height == 3


def test_select_keeps_best_duplicate_text():
    rows = [
        sentence_row(text='duplicated text about the drug ok', cue=False, label='A'),
        sentence_row(text='duplicated text about the drug ok', cue=True, label='B', pub='2', pmid='2'),
    ]
    out = select_sentences(sentence_frame(rows))
    assert out.height == 1
    assert out['label'].to_list() == ['B']


def two_doc_fixture():
    return sentence_frame([
        sentence_row(
            text='Alpha sentence from the abstract ok.', pub='5', pmid='5', year=2019, label='Aspirin', srank=1, h=1
        ),
        sentence_row(
            text='Beta sentence from the title ok ok ok.', pub='5', pmid='5', year=2019, label='ASA', srank=0, h=2
        ),
        sentence_row(
            text='Gamma sentence from a pmc paper ok.',
            pub='PMC7',
            pmid=None,
            pmcid='PMC7',
            year=2021,
            label='Aspirin',
            srank=1,
            h=3,
        ),
    ])


def test_request_text_matches_experiment_format():
    out = requests(select_sentences(two_doc_fixture()), NAMES)
    assert out.height == 1
    assert out['name'].to_list() == ['aspirin']
    assert out['request'].to_list() == [
        (
            'Drug: aspirin\n'
            'Names of the drug in these sentences: ASA, Aspirin\n'
            '\n'
            'Trial ID: pmc7\n'
            'Publication year: 2021\n'
            'Sentences:\n'
            'Gamma sentence from a pmc paper ok.\n'
            '---\n'
            'Trial ID: pmid5\n'
            'Publication year: 2019\n'
            'Sentences:\n'
            'Beta sentence from the title ok ok ok.\n'
            'Alpha sentence from the abstract ok.'
        )
    ]


def test_names_line_sorted_unique():
    rows = [
        sentence_row(text='first sentence about the drug ok ok', label='Zed', h=1),
        sentence_row(text='second sentence about the drug ok ok', label='Alpha', h=2),
        sentence_row(text='third sentence about the drug ok ok', label='Zed', h=3),
    ]
    out = requests(select_sentences(sentence_frame(rows)), NAMES)
    assert out['request'][0].splitlines()[1] == 'Names of the drug in these sentences: Alpha, Zed'


def test_pmc_only_doc_id_lowercase():
    out = requests(select_sentences(two_doc_fixture()), NAMES)
    assert {d['doc'] for d in out['docs'][0]} == {'pmid5', 'pmc7'}
    assert {s['doc'] for s in out['sentences'][0]} == {'pmid5', 'pmc7'}


def test_docs_and_sentences_columns():
    out = requests(select_sentences(two_doc_fixture()), NAMES)
    docs = {d['doc']: d['text'] for d in out['docs'][0]}
    assert docs['pmid5'] == 'Beta sentence from the title ok ok ok.\nAlpha sentence from the abstract ok.'
    sentences = out['sentences'][0]
    assert [(s['doc'], s['sha256']) for s in sentences] == sorted((s['doc'], s['sha256']) for s in sentences)
    gamma = next(s for s in sentences if s['doc'] == 'pmc7')
    assert gamma['sha256'] == hashlib.sha256(b'Gamma sentence from a pmc paper ok.').hexdigest()
    assert gamma['cue'] is False


def test_drug_without_sentences_gets_no_request():
    rows = [sentence_row(text='too short'), sentence_row(parent='CHEMBL2', text='a long enough sentence for drug two')]
    out = requests(select_sentences(sentence_frame(rows)), NAMES)
    assert out['parent'].to_list() == ['CHEMBL2']


def test_drug_without_a_name_gets_no_request():
    rows = [sentence_row(parent='CHEMBL9', text='a long enough sentence for drug nine')]
    assert requests(select_sentences(sentence_frame(rows)), NAMES).height == 0


def test_content_key_ignores_drug_name_and_order():
    a = {'doc': 'pmid1', 'sha256': 'aa', 'cue': True}
    b = {'doc': 'pmid2', 'sha256': 'bb', 'cue': False}
    key = content_key('CHEMBL1', [a, b])
    assert key == content_key('CHEMBL1', [b, a])
    assert key == 'CHEMBL1|' + hashlib.sha256(json.dumps([['pmid1', 'aa'], ['pmid2', 'bb']]).encode()).hexdigest()
    assert key != content_key('CHEMBL2', [a, b])
    renamed = requests(select_sentences(two_doc_fixture()), NAMES.with_columns(name=pl.lit('Renamed')))
    original = requests(select_sentences(two_doc_fixture()), NAMES)
    assert renamed['content_key'].to_list() == original['content_key'].to_list()


def test_null_year_doc_excluded_from_request_sentences_and_key():
    rows = [
        sentence_row(text='dated sentence from publication one ok', pub='1', pmid='1', year=2020),
        sentence_row(text='undated sentence from publication two', pub='2', pmid='2', year=None, label='Other'),
    ]
    out = requests(select_sentences(sentence_frame(rows)), NAMES)
    assert out['request'].to_list() == [
        (
            'Drug: aspirin\nNames of the drug in these sentences: Drug, Other\n\n'
            'Trial ID: pmid1\nPublication year: 2020\nSentences:\ndated sentence from publication one ok'
        )
    ]
    assert [d['doc'] for d in out['docs'][0]] == ['pmid1']
    assert [s['doc'] for s in out['sentences'][0]] == ['pmid1']
    only_dated = requests(select_sentences(sentence_frame(rows[:1])), NAMES)
    assert out['content_key'].to_list() == only_dated['content_key'].to_list()


def test_drug_with_only_null_year_docs_gets_no_request():
    rows = [
        sentence_row(text='undated sentence from publication two', year=None),
        sentence_row(parent='CHEMBL2', text='dated sentence for the other drug ok'),
    ]
    out = requests(select_sentences(sentence_frame(rows)), NAMES)
    assert out['parent'].to_list() == ['CHEMBL2']


def test_doc_without_an_id_excluded_from_request_sentences_and_key():
    rows = [
        sentence_row(text='identified sentence from publication one', pub='1', pmid='1'),
        sentence_row(text='anonymous sentence without any id here', pub=None, pmid=None, pmcid=None, label='Other'),
    ]
    out = requests(select_sentences(sentence_frame(rows)), NAMES)
    assert out['request'].to_list() == [
        (
            'Drug: aspirin\nNames of the drug in these sentences: Drug, Other\n\n'
            'Trial ID: pmid1\nPublication year: 2020\nSentences:\nidentified sentence from publication one'
        )
    ]
    assert [d['doc'] for d in out['docs'][0]] == ['pmid1']
    assert [s['doc'] for s in out['sentences'][0]] == ['pmid1']
    only_identified = requests(select_sentences(sentence_frame(rows[:1])), NAMES)
    assert out['content_key'].to_list() == only_identified['content_key'].to_list()


def test_drug_with_only_docs_without_an_id_gets_no_request():
    rows = [sentence_row(text='anonymous sentence without any id here', pub=None, pmid=None, pmcid=None)]
    assert requests(select_sentences(sentence_frame(rows)), NAMES).height == 0


def test_sentences_of_a_doc_keep_picked_order_on_a_section_rank_tie():
    rows = [
        sentence_row(text='aaa plain sentence of the publication ok', h=1, srank=1),
        sentence_row(text='zzz cue sentence of the publication ok', h=2, srank=1, cue=True),
    ]
    out = requests(select_sentences(sentence_frame(rows)), NAMES)
    assert (
        out['docs'][0][0]['text'] == 'zzz cue sentence of the publication ok\naaa plain sentence of the publication ok'
    )


def test_sentence_hash_is_pinned():
    """The tie-break hash of three fixed sentences is unchanged.

    Selection orders sentences by ``pl.col('text').hash(seed=HASH_SEED)``, which polars does not keep
    stable across versions. If this fails after a polars upgrade, the selection of drugs may change:
    bump ``logic_version`` of ``MOA_LITERATURE`` and re-plan the next run (drugs to compute, seed
    coverage) before updating the expected values.
    """
    texts = ['Sotorasib is a selective KRAS G12C inhibitor.', 'Aspirin inhibits COX-1 and COX-2.', '']
    hashed = pl.DataFrame({'text': texts}).select(h=pl.col('text').hash(seed=literature.HASH_SEED))['h']
    assert literature.HASH_SEED == 7
    assert hashed.to_list() == [3710146281297193471, 15559522047434505348, 7189315173487520619]
