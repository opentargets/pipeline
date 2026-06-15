"""PySpark implementation of the SearchEBI step.

Ported from platform-etl-backend SearchEBI step.
"""

import pyspark.sql.functions as f
from pyspark.sql import DataFrame

from pts.pyspark.common.session import Session


def _generate_datasets(
    diseases: DataFrame,
    targets: DataFrame,
    associations: DataFrame,
    evidence: DataFrame,
) -> dict[str, DataFrame]:
    """Join associations and evidence with target and disease metadata."""
    assoc_ds = (
        associations.join(targets, 'targetId', 'inner')
        .join(diseases, 'diseaseId', 'inner')
        .select('targetId', 'diseaseId', 'approvedSymbol', 'name', f.col('associationScore').alias('score'))
    )
    evidence_ds = (
        evidence.join(targets, 'targetId', 'inner')
        .join(diseases, 'diseaseId', 'inner')
        .select('targetId', 'diseaseId', 'approvedSymbol', 'name', 'score')
    )
    return {'associations': assoc_ds, 'evidence': evidence_ds}


def search_ebi(source: dict[str, str], destination: dict[str, str], settings, properties) -> None:
    """Entry point for the search_ebi step."""
    session = Session(app_name='search_ebi', properties=properties)
    spark = session.spark

    diseases = spark.read.parquet(source['disease']).withColumnRenamed('id', 'diseaseId')
    targets = spark.read.parquet(source['target']).withColumnRenamed('id', 'targetId')
    associations = spark.read.parquet(source['association'])
    evidence = spark.read.parquet(source['evidence'])

    datasets = _generate_datasets(diseases, targets, associations, evidence)

    datasets['associations'].write.mode('overwrite').parquet(destination['associations'])
    datasets['evidence'].write.mode('overwrite').parquet(destination['evidence'])
