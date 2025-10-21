import pyspark.sql.functions as f


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
    df2 = df.withColumn('gene', f.when(f.col('gene') == 'MLL', f.lit('KMT2A')).otherwise(f.col('gene')))
    assert {'KMT2A', 'TP53'} == {r['gene'] for r in df2.select('gene').collect()}


def test_create_dataframe_and_schema(spark):
    # create a small DF using the raw SparkSession
    rows = [('A', 1), ('B', 2)]
    df = spark.createDataFrame(rows, schema=['name', 'n'])
    assert df.count() == 2
    assert set(df.columns) == {'name', 'n'}
