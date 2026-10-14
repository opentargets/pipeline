"""Tests for the optional features of the Session that need jars: hail, the enhanced BGZF codec and Delta Lake.

The configuration tests never start the JVM. The jar tests do, each in a subprocess: jars are only loaded
when a JVM starts, and the tests share one JVM for their spark session.
"""

from __future__ import annotations

import gzip
import struct
import subprocess
import sys
import textwrap
import urllib.request
import zlib
from typing import TYPE_CHECKING, Any
from unittest import mock
from unittest.mock import MagicMock

import pytest
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

from pps.session import (
    BGZIP_CODEC,
    DELTA_CATALOG,
    DELTA_EXTENSION,
    HAIL_KRYO_REGISTRATOR,
    Session,
)

if TYPE_CHECKING:
    from pathlib import Path

BGZIP_JARS = [
    'https://repo1.maven.org/maven2/org/seqdoop/hadoop-bam/7.10.0/hadoop-bam-7.10.0.jar',
    'https://repo1.maven.org/maven2/com/github/samtools/htsjdk/2.13.2/htsjdk-2.13.2.jar',
]
"""The jars the clusters provide for the enhanced BGZF codec (see the `bgzip` jars of the staging dags)."""


def _session() -> Session:
    """A Session whose spark session is never started."""
    with mock.patch.object(SparkSession.Builder, 'getOrCreate', return_value=MagicMock(spec=SparkSession)):
        return Session(dynamic_allocation=False)


def _loader(session: Session) -> Any:
    """The mocked class loader `Session._check_jars` loads classes with."""
    spark: Any = session.spark
    return spark.sparkContext._jvm.java.lang.Thread.currentThread().getContextClassLoader()


def _conf(spark_submit: bool, monkeypatch: pytest.MonkeyPatch, **features: bool) -> dict[str, str]:
    """The configuration of a Session requesting `features`, in a JVM started by spark-submit or not."""
    if spark_submit:
        monkeypatch.setenv('PYSPARK_GATEWAY_PORT', '12345')
    else:
        monkeypatch.delenv('PYSPARK_GATEWAY_PORT', raising=False)
    flags = {'start_hail': False, 'use_enhanced_bgzip_codec': False, 'add_delta_lake': False} | features
    conf = _session()._build_config(dynamic_allocation=False, add_s3_connector=False, add_gcs_connector=False, **flags)
    return dict(conf.getAll())


@pytest.mark.no_shared_spark
class TestJarConfiguration:
    def test_delta(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conf = _conf(spark_submit=False, monkeypatch=monkeypatch, add_delta_lake=True)

        assert conf['spark.jars.packages'] == 'io.delta:delta-spark_2.12:3.2.0'
        assert conf['spark.sql.extensions'] == DELTA_EXTENSION
        assert conf['spark.sql.catalog.spark_catalog'] == DELTA_CATALOG

    def test_delta_keeps_other_extensions(self) -> None:
        conf = SparkConf(loadDefaults=False).set('spark.sql.extensions', 'some.Extension')
        conf = Session._setup_delta_config(Session._setup_delta_config(conf))

        assert conf.get('spark.sql.extensions') == f'some.Extension,{DELTA_EXTENSION}'

    def test_bgzip(self, monkeypatch: pytest.MonkeyPatch) -> None:
        conf = _conf(spark_submit=False, monkeypatch=monkeypatch, use_enhanced_bgzip_codec=True)

        assert conf['spark.jars.packages'] == 'org.seqdoop:hadoop-bam:7.10.0'
        assert conf['spark.hadoop.io.compression.codecs'] == BGZIP_CODEC

    @pytest.mark.parametrize(
        'feature', ['add_delta_lake', 'use_enhanced_bgzip_codec', 'start_hail'], ids=['delta', 'bgzip', 'hail']
    )
    def test_spark_submit_jvm_gets_jars_from_its_cluster(self, monkeypatch: pytest.MonkeyPatch, feature: str) -> None:
        conf = _conf(spark_submit=True, monkeypatch=monkeypatch, **{feature: True})

        assert 'spark.jars.packages' not in conf
        assert 'spark.jars' not in conf
        assert 'spark.driver.extraClassPath' not in conf
        # their use is still configured
        assert {DELTA_EXTENSION, BGZIP_CODEC, HAIL_KRYO_REGISTRATOR} & set(conf.values())


class TestCheckJars:
    @staticmethod
    def _session_missing(class_name: str) -> Session:
        session = _session()
        loader = _loader(session)

        def load_class(name: str) -> MagicMock:
            if name == class_name:
                raise RuntimeError(f'ClassNotFoundException: {name}')
            return MagicMock(**{'getPackage.return_value.getImplementationVersion.return_value': None})

        loader.loadClass.side_effect = load_class
        return session

    def test_missing_jar_on_dataproc(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv('PYSPARK_GATEWAY_PORT', '12345')
        session = self._session_missing(DELTA_EXTENSION)

        with pytest.raises(RuntimeError, match='list `delta` in the jars of the cluster'):
            session._check_jars(start_hail=False, use_enhanced_bgzip_codec=False, add_delta_lake=True)

    def test_unrequested_features_are_not_checked(self) -> None:
        session = self._session_missing(DELTA_EXTENSION)

        session._check_jars(start_hail=False, use_enhanced_bgzip_codec=False, add_delta_lake=False)

    def test_delta_jar_must_match_the_package_minor_version(self) -> None:
        session = _session()
        loader = _loader(session)
        loader.loadClass.return_value.getPackage.return_value.getImplementationVersion.return_value = '3.3.0'

        with pytest.raises(RuntimeError, match=r'does not match the delta-spark package \(3\.2\.0\)'):
            session._check_jars(start_hail=False, use_enhanced_bgzip_codec=False, add_delta_lake=True)


def test_load_data_delta() -> None:
    session = _session()

    session.load_data('some/table', fmt='delta')

    spark: Any = session.spark
    spark.read.load.assert_called_once_with('some/table', format='delta', schema=None)


def _run(script: str) -> None:
    """Run a python script in a fresh interpreter, so it starts a JVM of its own."""
    result = subprocess.run(
        [sys.executable, '-c', textwrap.dedent(script)], capture_output=True, text=True, timeout=600, check=False
    )
    assert result.returncode == 0, result.stderr[-5000:]


def test_delta_lake(tmp_path: Path) -> None:
    """A delta table can be written, read and merged into, with the jar the Session adds itself."""
    _run(f"""
        from delta.tables import DeltaTable
        from pps.session import Session

        session = Session('local[1]', add_delta_lake=True, dynamic_allocation=False)
        spark = session.spark
        path = '{tmp_path / 'table'}'

        spark.createDataFrame([(1, 'a'), (2, 'b')], 'id int, value string').write.format('delta').save(path)
        updates = spark.createDataFrame([(2, 'B'), (3, 'c')], 'id int, value string')
        (
            DeltaTable.forPath(spark, path).alias('t')
            .merge(updates.alias('u'), 't.id = u.id')
            .whenMatchedUpdateAll()
            .whenNotMatchedInsertAll()
            .execute()
        )

        rows = sorted(tuple(r) for r in session.load_data(path, fmt='delta').collect())
        assert rows == [(1, 'a'), (2, 'B'), (3, 'c')], rows
    """)


def _bgzf(data: bytes) -> bytes:
    """Block gzip `data` in a single BGZF block, followed by the BGZF end of file block."""

    def block(payload: bytes) -> bytes:
        deflated = gzip.compress(payload, mtime=0)[10:-8]  # raw deflate stream of a gzip member
        # gzip header with the BC extra subfield holding the total block size minus 1
        header = b'\x1f\x8b\x08\x04\x00\x00\x00\x00\x00\xff\x06\x00BC\x02\x00'
        size = len(header) + 2 + len(deflated) + 8
        crc = zlib.crc32(payload)
        return header + struct.pack('<H', size - 1) + deflated + struct.pack('<II', crc, len(payload))

    return block(data) + block(b'')


def test_bgzip_jars(tmp_path: Path) -> None:
    """The jars the clusters list for `bgzip` are enough to read block gzipped files through the codec."""
    jars = []
    for url in BGZIP_JARS:
        jar = tmp_path / url.rsplit('/', 1)[-1]
        urllib.request.urlretrieve(url, jar)  # noqa: S310
        jars.append(jar.as_posix())
    # hadoop picks a codec by extension, and the only codec configured reads .gz
    data = tmp_path / 'data.tsv.gz'
    data.write_bytes(_bgzf(b'a\tb\n1\t2\n3\t4\n'))

    # the jars as a cluster provides them: in spark.jars, without resolving their dependencies
    _run(f"""
        from pyspark.sql import SparkSession

        spark = (
            SparkSession.Builder()
            .master('local[1]')
            .config('spark.jars', '{','.join(jars)}')
            .config('spark.hadoop.io.compression.codecs', '{BGZIP_CODEC}')
            .getOrCreate()
        )
        df = spark.read.csv('{data}', sep='\\t', header=True)
        assert [tuple(r) for r in df.collect()] == [('1', '2'), ('3', '4')], df.collect()
        loader = spark.sparkContext._jvm.java.lang.Thread.currentThread().getContextClassLoader()
        loader.loadClass('{BGZIP_CODEC}')
    """)
