"""End-to-end tests: how the disease index resolves chains of obsoleted terms.

Runs the full ``disease`` transformer on a tiny OBO-graph JSON so the
assertions cover the published ``obsoleteTerms`` / ``obsoleteXRefs`` columns,
not just the IAO_0100001 rewriting done by ``resolve_replacement_chains``.
See opentargets/issues#2929.
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from pts.transformers.disease.coalescing import _IAO_REPLACED_BY
from pts.transformers.disease.disease import disease

EFO = 'http://www.ebi.ac.uk/efo/'


def _url(short_id: str) -> str:
    return f'{EFO}{short_id}'


def _node(
    short_id: str,
    *,
    lbl: str | None = None,
    deprecated: bool | None = None,
    replaced_by: str | None = None,
    xrefs: list[str] | None = None,
) -> dict:
    bpv = [{'pred': _IAO_REPLACED_BY, 'val': replaced_by}] if replaced_by else []
    return {
        'id': _url(short_id),
        'lbl': lbl or f'label of {short_id}',
        'type': 'CLASS',
        'meta': {
            'basicPropertyValues': bpv,
            'comments': [],
            'definition': {'val': None, 'xrefs': []},
            'deprecated': deprecated,
            'subsets': [],
            'synonyms': [],
            'xrefs': [{'val': x} for x in (xrefs or [])],
        },
    }


def _is_a(child: str, parent: str) -> dict:
    return {'sub': _url(child), 'pred': 'is_a', 'obj': _url(parent)}


def _run(tmp_path: Path, nodes: list[dict], edges: list[dict]) -> pl.DataFrame:
    source = tmp_path / 'efo.json'
    source.write_text(json.dumps({'graphs': [{'nodes': nodes, 'edges': edges}]}))
    destination = tmp_path / 'out'
    disease(str(source), str(destination), {}, None)
    return pl.read_parquet(destination / '*.parquet')


def _row(index: pl.DataFrame, short_id: str) -> dict:
    rows = index.filter(pl.col('id') == short_id).to_dicts()
    assert len(rows) == 1, f'{short_id} not in the index'
    return rows[0]


ROOT = _node('EFO_root', lbl='disease')
# The synonym pivot expects every synonym predicate to occur somewhere, as it
# does in the real ontology.
ROOT['meta']['synonyms'] = [
    {'pred': pred, 'synonymType': None, 'val': f'root {pred}', 'xrefs': []}
    for pred in ('hasExactSynonym', 'hasRelatedSynonym', 'hasNarrowSynonym', 'hasBroadSynonym')
]


class TestObsoleteChains:
    def test_two_hop_chain_lands_on_the_live_term(self, tmp_path: Path) -> None:
        # A -(replaced by)-> B -(replaced by)-> C, only C is live.
        index = _run(
            tmp_path,
            [
                ROOT,
                _node('EFO_A', deprecated=True, replaced_by=_url('EFO_B'), xrefs=['MONDO:A']),
                _node('EFO_B', deprecated=True, replaced_by=_url('EFO_C'), xrefs=['MONDO:B']),
                _node('EFO_C', xrefs=['MONDO:C']),
            ],
            [_is_a('EFO_C', 'EFO_root')],
        )
        assert set(index['id']) == {'EFO_root', 'EFO_C'}
        c = _row(index, 'EFO_C')
        assert sorted(c['obsoleteTerms']) == ['EFO_A', 'EFO_B']
        assert sorted(c['obsoleteXRefs']) == ['MONDO:A', 'MONDO:B']

    def test_four_hop_chain_lands_on_the_live_term(self, tmp_path: Path) -> None:
        index = _run(
            tmp_path,
            [
                ROOT,
                _node('EFO_A', deprecated=True, replaced_by=_url('EFO_B')),
                _node('EFO_B', deprecated=True, replaced_by=_url('EFO_C')),
                _node('EFO_C', deprecated=True, replaced_by=_url('EFO_D')),
                _node('EFO_D', deprecated=True, replaced_by=_url('EFO_E')),
                _node('EFO_E'),
            ],
            [_is_a('EFO_E', 'EFO_root')],
        )
        assert sorted(_row(index, 'EFO_E')['obsoleteTerms']) == ['EFO_A', 'EFO_B', 'EFO_C', 'EFO_D']

    def test_chain_through_a_coalesced_duplicate(self, tmp_path: Path) -> None:
        # The source obsoletes A into B, which is live in the source but shares
        # its label with C; the name-duplicate rule supersedes one of them, so
        # the source's single hop becomes a two-hop chain at build time.
        index = _run(
            tmp_path,
            [
                ROOT,
                _node('EFO_A', deprecated=True, replaced_by=_url('EFO_B')),
                _node('EFO_B', lbl='same disease'),
                _node('EFO_C', lbl='same disease'),
            ],
            [_is_a('EFO_B', 'EFO_root'), _is_a('EFO_C', 'EFO_root')],
        )
        survivors = {'EFO_B', 'EFO_C'} & set(index['id'])
        assert len(survivors) == 1
        survivor = survivors.pop()
        loser = ({'EFO_B', 'EFO_C'} - {survivor}).pop()
        assert sorted(_row(index, survivor)['obsoleteTerms']) == sorted(['EFO_A', loser])

    def test_chain_ending_in_an_obsolete_term_without_replacement_is_lost(self, tmp_path: Path) -> None:
        # Documented limitation: B is obsolete and names no successor, so there
        # is no live term to attach A or B to.
        index = _run(
            tmp_path,
            [
                ROOT,
                _node('EFO_A', deprecated=True, replaced_by=_url('EFO_B')),
                _node('EFO_B', deprecated=True),
                _node('EFO_C'),
            ],
            [_is_a('EFO_C', 'EFO_root')],
        )
        assert set(index['id']) == {'EFO_root', 'EFO_C'}
        assert all(terms.is_empty() for terms in index['obsoleteTerms'])

    @pytest.mark.parametrize('pointer', ['EFO:EFO_C', 'EFO_C'])
    def test_pointer_not_written_as_a_node_url_is_lost(self, tmp_path: Path, pointer: str) -> None:
        # Documented limitation: obsolete_ids joins the pointer against the
        # node's full URL, so a CURIE / bare id pointer reaches nothing -- even
        # a single hop, and even though the target is live.
        index = _run(
            tmp_path,
            [
                ROOT,
                _node('EFO_A', deprecated=True, replaced_by=_url('EFO_B')),
                _node('EFO_B', deprecated=True, replaced_by=pointer),
                _node('EFO_C'),
            ],
            [_is_a('EFO_C', 'EFO_root')],
        )
        assert _row(index, 'EFO_C')['obsoleteTerms'] == []
