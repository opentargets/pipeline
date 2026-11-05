from __future__ import annotations

from typing import TYPE_CHECKING, Any
from urllib.parse import parse_qs, urlparse

import polars as pl
import pytest
from otter.config.model import Config

from pts.transformers.clinical_mira.moa import papers

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

URL = 'https://example.org/search'


def _config(tmp_path: Path) -> Config:
    return Config(step='test', steps=['test'], work_path=tmp_path / 'work')


@pytest.fixture(autouse=True)
def _no_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(papers.time, 'sleep', lambda _s: None)


def test_trial_papers_keeps_result_and_derived(monkeypatch: pytest.MonkeyPatch) -> None:
    table = pl.DataFrame({
        'nct_id': ['NCT1', 'NCT1', 'NCT2', 'NCT3', 'NCT4', 'NCT5'],
        'pmid': ['11', '11', '22', '33', None, 'x1'],
        'reference_type': ['RESULT', 'RESULT', 'DERIVED', 'BACKGROUND', 'RESULT', 'RESULT'],
    })
    seen: dict[str, Any] = {}

    def fake(archive: str, tables: list[str], scratch_root: str) -> dict[str, pl.DataFrame]:
        seen['args'] = (archive, tables, scratch_root)
        return {'study_references': table}

    monkeypatch.setattr(papers, 'read_aact_tables', fake)
    out = papers.trial_papers('a.zip', '/s', ['DERIVED', 'RESULT'])
    assert seen['args'] == ('a.zip', ['study_references'], '/s')
    assert out.columns == ['nct_id', 'pmid']
    assert sorted(out.rows()) == [('NCT1', '11'), ('NCT2', '22')]


def test_select_papers_newest_first_capped() -> None:
    trials = pl.DataFrame({'nct_id': ['N1'], 'drug_name': ['d']})
    links = pl.DataFrame({'nct_id': ['N1'] * 7, 'pmid': ['5', '100', '20', '3', '9', '40', '7']})
    out = papers.select_papers(trials, links, 5)
    assert out.columns == ['drug_name', 'pmid', 'nct_id']
    assert out['pmid'].to_list() == ['100', '40', '20', '9', '7']


def test_select_papers_one_entry_per_pool_and_one_per_each_pool() -> None:
    trials = pl.DataFrame({'nct_id': ['N2', 'N1', 'N3'], 'drug_name': ['a', 'a', 'b']})
    links = pl.DataFrame({'nct_id': ['N1', 'N2', 'N3'], 'pmid': ['10', '10', '10']})
    out = papers.select_papers(trials, links, 5).sort('drug_name')
    assert out.rows() == [('a', '10', 'N1'), ('b', '10', 'N3')]


def test_paper_id() -> None:
    assert papers.paper_id('39167366') == 'pmid39167366'


def test_paper_text_cleans_html_and_truncates() -> None:
    text = papers.paper_text('7', ' A <i>title</i> ', '<h4>Background</h4>&amp;  more\ntext ' + 'x' * 50, 20)
    assert text == 'Trial ID: pmid7\nOfficial Title: A title\nDescription: Background & more te'


def _pmids(url: str) -> list[str]:
    query = parse_qs(urlparse(url).query)['query'][0]
    return query.removeprefix('SRC:MED AND (').removesuffix(')').replace('EXT_ID:', '').split(' OR ')


def _fake_get(calls: list[str], fail: str | None = None) -> Callable[[str], dict[str, Any]]:
    def get(url: str) -> dict[str, Any]:
        calls.append(url)
        ids = _pmids(url)
        if ids[0] == fail:
            msg = 'down'
            raise OSError(msg)
        return {
            'resultList': {'result': [{'pmid': p, 'title': f't{p}', 'abstractText': f'abs {p}'} for p in ids]},
        }

    return get


def test_fetch_abstracts_chunks_of_40() -> None:
    calls: list[str] = []
    out = papers.fetch_abstracts([str(i) for i in range(1, 91)], URL, _fake_get(calls))
    assert [len(_pmids(c)) for c in calls] == [40, 40, 10]
    q = parse_qs(urlparse(calls[0]).query)
    assert calls[0].startswith(URL + '?')
    assert q['resultType'] == ['core']
    assert q['format'] == ['json']
    assert q['pageSize'] == ['100']
    assert q['query'][0].startswith('SRC:MED AND (EXT_ID:1 OR EXT_ID:2')
    assert out.columns == ['pmid', 'title', 'abstract']
    assert out.height == 90


def test_fetch_abstracts_skips_failed_chunk() -> None:
    calls: list[str] = []
    out = papers.fetch_abstracts([str(i) for i in range(1, 91)], URL, _fake_get(calls, fail='41'))
    assert len(calls) == 2 + 3
    assert set(out['pmid']) == {str(i) for i in range(1, 91)} - {str(i) for i in range(41, 81)}


def test_fetch_abstracts_drops_empty_abstracts() -> None:
    def get(_url: str) -> dict[str, Any]:
        return {
            'resultList': {
                'result': [
                    {'pmid': '1', 'title': 't', 'abstractText': 'a'},
                    {'pmid': '2', 'title': 't', 'abstractText': None},
                    {'pmid': '3', 'title': 't', 'abstractText': '  '},
                    {'pmid': '4', 'title': 't'},
                ],
            },
        }

    out = papers.fetch_abstracts(['1', '2', '3', '4'], URL, get)
    assert out['pmid'].to_list() == ['1']


def test_fetch_abstracts_without_pmids_makes_no_call() -> None:
    calls: list[str] = []
    out = papers.fetch_abstracts([], URL, _fake_get(calls))
    assert calls == []
    assert out.columns == ['pmid', 'title', 'abstract']
    assert out.height == 0


def test_abstract_store_round_trip_latest_wins(tmp_path: Path) -> None:
    config = _config(tmp_path)
    store = str(tmp_path / 'store')
    empty = papers.read_abstracts(store, config)
    assert empty.height == 0
    assert empty.columns == ['pmid', 'title', 'abstract', 'fetched_at']
    first = pl.DataFrame({'pmid': ['1', '2'], 'title': ['t1', 't2'], 'abstract': ['old1', 'a2']})
    second = pl.DataFrame({'pmid': ['1'], 'title': ['t1b'], 'abstract': ['new1']})
    papers.write_abstracts(first, store, 'run-a', config)
    papers.write_abstracts(second, store, 'run-b', config)
    out = papers.read_abstracts(store, config).sort('pmid')
    assert out.select('pmid', 'title', 'abstract').rows() == [('1', 't1b', 'new1'), ('2', 't2', 'a2')]
