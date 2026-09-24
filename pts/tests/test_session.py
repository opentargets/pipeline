import os
import subprocess
import sys
from textwrap import dedent

import pyspark.sql.functions as f
import pytest

from pts.pyspark.common.session import Session


@pytest.mark.slow
def test_load_csv_and_replace(tmp_path, pts_session):
    # Prepare a tiny TSV/CSV file
    p = tmp_path / 'small.tsv'
    p.write_text('gene\tvalue\nTP53\t1\nMLL\t2\n')

    # Use the Session wrapper to load it (note: options passed to load_data)
    df = pts_session.load_data(str(p), format='csv', header=True, sep='\t')

    # Basic assertions
    assert df.count() == 2
    names = {r['gene'] for r in df.select('gene').collect()}
    assert names == {'TP53', 'MLL'}

    # Example transform: replace 'MLL' -> 'KMT2A' inline (runtime check)
    df2 = df.withColumn(
        'gene',
        f.when(f.col('gene') == 'MLL', f.lit('KMT2A')).otherwise(f.col('gene')),
    )
    assert {'KMT2A', 'TP53'} == {r['gene'] for r in df2.select('gene').collect()}


@pytest.mark.slow
def test_create_dataframe_and_schema(spark):
    # create a small DF using the raw SparkSession
    rows = [('A', 1), ('B', 2)]
    df = spark.createDataFrame(rows, schema=['name', 'n'])
    assert df.count() == 2
    assert set(df.columns) == {'name', 'n'}


def test_merge_jars_packages():
    assert Session._merge_jars_packages(None, None) is None
    assert Session._merge_jars_packages('a:b:1', None) == 'a:b:1'
    assert Session._merge_jars_packages(None, 'c:d:2') == 'c:d:2'
    assert Session._merge_jars_packages('a:b:1', 'c:d:2') == 'a:b:1,c:d:2'
    assert Session._merge_jars_packages('a:b:1,c:d:2', 'c:d:2,e:f:3') == 'a:b:1,c:d:2,e:f:3'


def test_merge_jars_packages_explicit_version_overrides_default():
    assert (
        Session._merge_jars_packages(
            'com.johnsnowlabs.nlp:spark-nlp_2.12:6.1.5,base:other:1',
            'com.johnsnowlabs.nlp:spark-nlp_2.12:6.2.0,custom:package:2',
        )
        == 'com.johnsnowlabs.nlp:spark-nlp_2.12:6.2.0,base:other:1,custom:package:2'
    )


@pytest.mark.parametrize('installed_version', ['6.1.5', '6.2.0'])
def test_session_local_config_contains_sparknlp_and_gcs(monkeypatch, installed_version):
    def package_version(package):
        assert package == 'spark-nlp'
        return installed_version

    monkeypatch.setattr('pts.pyspark.common.session.version', package_version)
    # Pure config test via _effective_properties (isolated from JVM global SparkConf)
    s = Session.__new__(Session)
    s.is_dataproc = False
    eff = s._effective_properties({})
    jars = eff.get('spark.jars.packages')
    assert jars is not None and 'gcs-connector' in jars
    assert f'com.johnsnowlabs.nlp:spark-nlp_2.12:{installed_version}' in jars.split(',')
    # caller-supplied jars are merged, not dropped
    eff2 = s._effective_properties({'spark.jars.packages': 'my.org:custom:1.0'})
    jars2 = eff2.get('spark.jars.packages')
    assert jars2 is not None and 'my.org:custom:1.0' in jars2
    assert 'gcs-connector' in jars2
    assert f'com.johnsnowlabs.nlp:spark-nlp_2.12:{installed_version}' in jars2.split(',')


def test_session_dataproc_does_not_force_jars():
    session = Session.__new__(Session)
    session.is_dataproc = True

    assert session._effective_properties({}).get('spark.jars.packages') is None
    assert (
        session._effective_properties({'spark.jars.packages': 'my.org:custom:1.0'})['spark.jars.packages']
        == 'my.org:custom:1.0'
    )


@pytest.mark.slow
@pytest.mark.spark_nlp
def test_ontoma_spark_nlp_is_available():
    """Start an isolated local JVM so the shared test session stays lightweight."""
    # This performs real Maven/Ivy resolution and is intentionally excluded from
    # routine CI; use ``pytest -m spark_nlp`` to run it explicitly.
    env = os.environ.copy()
    for key in ('PYSPARK_GATEWAY_PORT', 'PYSPARK_GATEWAY_SECRET', 'DATAPROC_CLUSTER_NAME'):
        env.pop(key, None)

    result = subprocess.run(
        [
            sys.executable,
            '-c',
            dedent("""
                from ontoma import OnToma
                from pts.pyspark.common.session import Session

                session = Session(app_name='pts-spark-nlp-test', spark_uri='local[1]')
                try:
                    assert OnToma._spark_nlp_available(session.spark)
                finally:
                    session.stop()
            """),
        ],
        env=env,
        capture_output=True,
        text=True,
        timeout=300,
    )
    assert result.returncode == 0, result.stdout + result.stderr
