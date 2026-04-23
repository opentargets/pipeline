"""Literature word vector index generation.

Ported from Vectors.scala in platform-etl-backend.
Loads a trained Word2Vec model, extracts word vectors, computes L2 norms,
assigns entity categories, and writes the result as a single-partition parquet.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
from pyspark.ml.feature import Word2VecModel
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import Vectors
from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql.types import DoubleType

from pts.pyspark.common.session import Session

_OUTPUT_COLUMNS = ['category', 'word', 'norm', 'vector']


def _compute_vectors(vectors: DataFrame) -> DataFrame:
    """Transform raw Word2Vec vectors: assign category, compute norm, serialize.

    Args:
        vectors: DataFrame with columns [word, vector] from Word2VecModel.getVectors().

    Returns:
        DataFrame with columns [category, word, norm, vector].
    """
    norm_udf = f.udf(lambda v: float(Vectors.norm(v, 2.0)), DoubleType())

    return (
        vectors
        .withColumn(
            'category',
            f
            .when(f.col('word').startswith('ENSG'), f.lit('target'))
            .when(f.col('word').startswith('CHEMBL'), f.lit('drug'))
            .otherwise(f.lit('disease')),
        )
        .withColumn('norm', norm_udf(f.col('vector')))
        .withColumn('vector', vector_to_array(f.col('vector')))
        .select(*_OUTPUT_COLUMNS)
    )


def literature_vector(
    source: dict[str, str] | str,
    destination: dict[str, str] | str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate vector index from trained Word2Vec model."""
    _spark = Session(app_name='literature_vector', properties=properties).spark

    model_path = source['model'] if isinstance(source, dict) else source
    logger.info(f'Loading Word2Vec model from {model_path}')
    model = Word2VecModel.load(model_path)
    raw_vectors = model.getVectors()

    logger.info('Computing vector index')
    result = _compute_vectors(raw_vectors).coalesce(1)

    dest = destination['vectors'] if isinstance(destination, dict) else destination
    logger.info(f'Writing vector index to {dest}')
    result.write.mode('overwrite').parquet(dest)
