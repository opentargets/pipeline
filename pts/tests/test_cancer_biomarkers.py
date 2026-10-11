"""Tests for the cancer_biomarkers pyspark module."""

from pyspark.sql import functions as f
from pyspark.sql.types import ArrayType, StringType

from pts.pyspark.cancer_biomarkers import CancerBiomarkersEvidenceGenerator


def _zip_alterations_with_type(alterations, alteration_type):
    """The former `zip_alterations_with_type` UDF."""
    alteration_types = alteration_type * len(alterations)
    return list(zip(alterations, alteration_types, strict=False))


def test_zip_alterations_with_type_matches_udf(spark):
    cases = [
        (['MET:Y1230C', 'Y1235D'], ['MUT']),
        (['a', 'b', 'c'], ['MUT', 'CNA']),
        (['a', 'b'], ['MUT', 'CNA', 'FUS']),
        (['a'], ['MUT', 'CNA']),
        (['a', 'a'], ['MUT', 'MUT']),
        (['a', 'b', 'c', 'd', 'e'], ['MUT', 'CNA']),
        ([''], ['']),
        ([], ['MUT']),
        (['a'], []),
        ([], []),
    ]
    df = spark.createDataFrame(cases, 'alterations ARRAY<STRING>, alteration_types ARRAY<STRING>')
    zip_udf = f.udf(_zip_alterations_with_type, ArrayType(ArrayType(StringType())))

    rows = df.select(
        CancerBiomarkersEvidenceGenerator.zip_alterations_with_type(
            f.col('alterations'), f.col('alteration_types')
        ).alias('native'),
        zip_udf(f.col('alterations'), f.col('alteration_types')).alias('udf'),
    ).collect()

    assert [r.native for r in rows] == [r.udf for r in rows]
    assert rows[1].native == [['a', 'MUT'], ['b', 'CNA'], ['c', 'MUT']]


def test_zip_alterations_with_type_null_yields_no_pairs(spark):
    df = spark.createDataFrame(
        [(None, ['MUT']), (['a'], None)], 'alterations ARRAY<STRING>, alteration_types ARRAY<STRING>'
    )
    exploded = df.select(
        f.explode(
            CancerBiomarkersEvidenceGenerator.zip_alterations_with_type(f.col('alterations'), f.col('alteration_types'))
        )
    )
    assert exploded.count() == 0
