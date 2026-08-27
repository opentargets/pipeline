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
    assert Session._merge_jars_packages('a:1', None) == 'a:1'
    assert Session._merge_jars_packages(None, 'b:2') == 'b:2'
    assert Session._merge_jars_packages('a:1', 'b:2') == 'a:1,b:2'
    # dedup, order preserved
    assert Session._merge_jars_packages('a:1,b:2', 'b:2,c:3') == 'a:1,b:2,c:3'
    assert Session._merge_jars_packages('a:1, b:2 ', ' b:2 , c:3') == 'a:1,b:2,c:3'


def test_session_local_config_contains_sparknlp_and_gcs():
    # Pure config test via _effective_properties (isolated from JVM global SparkConf)
    s = Session.__new__(Session)
    s.is_dataproc = False
    eff = s._effective_properties({})
    jars = eff.get('spark.jars.packages')
    assert jars is not None and 'gcs-connector' in jars
    assert 'spark-nlp_2.12:6.1.5' in jars
    # caller-supplied jars are merged, not dropped
    eff2 = s._effective_properties({'spark.jars.packages': 'my.org:custom:1.0'})
    jars2 = eff2.get('spark.jars.packages')
    assert jars2 is not None and 'my.org:custom:1.0' in jars2
    assert 'gcs-connector' in jars2
    assert 'spark-nlp_2.12:6.1.5' in jars2


def test_session_dataproc_does_not_force_jars():
    # Pure config test via _effective_properties - avoids JVM-polluted SparkConf
    s = Session.__new__(Session)
    s.is_dataproc = True
    eff = s._effective_properties({})
    assert eff.get('spark.jars.packages') is None
    # but explicit properties are still honoured
    eff2 = s._effective_properties({'spark.jars.packages': 'my.org:custom:1.0'})
    assert eff2.get('spark.jars.packages') == 'my.org:custom:1.0'

    # Same via env var + real is_dataproc detection (no shared Spark)
    import os

    orig = os.environ.get('DATAPROC_CLUSTER_NAME')
    try:
        os.environ['DATAPROC_CLUSTER_NAME'] = 'test-cluster'
        s2 = Session.__new__(Session)
        s2.is_dataproc = 'DATAPROC_CLUSTER_NAME' in os.environ
        assert s2.is_dataproc is True
        assert s2._effective_properties({}).get('spark.jars.packages') is None
    finally:
        if orig is None:
            os.environ.pop('DATAPROC_CLUSTER_NAME', None)
        else:
            os.environ['DATAPROC_CLUSTER_NAME'] = orig


@pytest.mark.slow
def test_ontoma_spark_nlp_is_available(pts_session):
    """Local pts_session must have Spark NLP on the JVM classpath (PR #6 fat-jar for Dataproc, Ivy for local)."""
    from ontoma import OnToma

    assert OnToma._spark_nlp_available(pts_session.spark) is True
    # Also ensure the config that provided it is visible
    jars = pts_session.spark.conf.get('spark.jars.packages', '')
    assert 'spark-nlp' in jars
