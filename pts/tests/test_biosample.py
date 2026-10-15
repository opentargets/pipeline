"""Tests for the biosample ancestor and descendant closure."""

from __future__ import annotations

import threading

import polars as pl

from pts.transformers.biosample import _ancestor_and_descendant_closure


def closure(parents: dict[str, list[str] | None]) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    direct = pl.DataFrame(
        {'biosampleId': list(parents), 'parents': list(parents.values())},
        schema={'biosampleId': pl.String, 'parents': pl.List(pl.String)},
    )
    result: list[tuple[pl.DataFrame, pl.DataFrame]] = []
    worker = threading.Thread(target=lambda: result.append(_ancestor_and_descendant_closure(direct)), daemon=True)
    worker.start()
    worker.join(timeout=10)
    assert result, 'closure did not finish'
    ancestors, descendants = result[0]
    return (
        {r['biosampleId']: sorted(r['ancestors']) for r in ancestors.iter_rows(named=True)},
        {r['biosampleId']: sorted(r['descendants']) for r in descendants.iter_rows(named=True)},
    )


def test_closure_follows_every_path() -> None:
    ancestors, descendants = closure({'a': None, 'b': ['a'], 'c': ['a'], 'd': ['b', 'c']})
    assert ancestors == {'b': ['a'], 'c': ['a'], 'd': ['a', 'b', 'c']}
    assert descendants == {'a': ['b', 'c', 'd'], 'b': ['d'], 'c': ['d']}


def test_closure_terminates_on_a_cycle() -> None:
    ancestors, _ = closure({'a': ['b'], 'b': ['a'], 'c': ['a']})
    assert ancestors == {'a': ['a', 'b'], 'b': ['a', 'b'], 'c': ['a', 'b']}
