"""Tests for the intogen pyspark module."""

from pyspark.sql import functions as f
from pyspark.sql import types as t

from pts.pyspark.intogen import DOI_TO_PMID_MAPPING, parse_source


def test_parse_source_matches_udf(spark):
    references = [
        'PMID: 23334666',
        'DOI: 10.1038/ng.2529',
        'DOI:10.1101/162784 ',
        'DOI: 10.1038/unmapped',
        'DOI: 10.1038/NG.2529',
        'WEB: https://example.org',
        'no colon',
        'a: b: c',
        'empty:',
        '',
        None,
    ]
    df = spark.createDataFrame([(r,) for r in references], 'REFERENCE STRING')
    # the former DOI to PMID lookup
    map_udf = f.udf(lambda v: DOI_TO_PMID_MAPPING.get(v, v), t.StringType())
    processed = f.when(f.col('REFERENCE').startswith('WEB'), f.lit(None)).otherwise(
        f.trim(f.split(f.col('REFERENCE'), ':')[1])
    )
    expected = f.when(map_udf(processed).isNotNull(), f.array(map_udf(processed)))

    rows = df.select(parse_source(f.col('REFERENCE')).alias('native'), expected.alias('udf')).collect()

    assert [r.native for r in rows] == [r.udf for r in rows]
    assert rows[1].native == ['23334666']
    assert rows[2].native == ['32025007']
