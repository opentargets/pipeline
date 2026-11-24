"""Polars and the spark `Session` read and write the same delta tables."""

from __future__ import annotations

import subprocess
import sys
import textwrap
from typing import TYPE_CHECKING

import polars as pl

if TYPE_CHECKING:
    from pathlib import Path


def _spark(script: str, table: Path) -> None:
    """Run `script` with `session`, a delta lake `Session`, and `path`, the table, in a JVM of its own."""
    preamble = f"""
        from pps.session import Session

        session = Session('local[1]', add_delta_lake=True, dynamic_allocation=False)
        spark = session.spark
        path = '{table}'
    """
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(preamble) + textwrap.dedent(script)],
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-5000:]


def test_polars_and_spark_share_a_delta_table(tmp_path: Path) -> None:
    table = tmp_path / 'table'
    _spark(
        """
        from delta.tables import DeltaTable

        spark.createDataFrame([(1, 'a'), (2, 'b')], 'id int, value string').write.format('delta').save(path)
        updates = spark.createDataFrame([(2, 'B')], 'id int, value string')
        DeltaTable.forPath(spark, path).alias('t').merge(
            updates.alias('u'), 't.id = u.id'
        ).whenMatchedUpdateAll().execute()
        """,
        table,
    )

    assert pl.read_delta(str(table)).sort('id').rows() == [(1, 'a'), (2, 'B')]

    pl.DataFrame({'id': [3], 'value': ['c']}, schema={'id': pl.Int32, 'value': pl.String}).write_delta(
        str(table), mode='append'
    )
    _spark(
        """
        rows = sorted(tuple(r) for r in session.load_data(path, fmt='delta').collect())
        assert rows == [(1, 'a'), (2, 'B'), (3, 'c')], rows
        """,
        table,
    )
