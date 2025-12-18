"""Lookup tables used by baseline expression validation.

The target/biosample indexes that feed validation always come from the
canonical OT outputs (`output/target` and `output/biosample`) so we assume
their schemas (`id`, `proteinIds`, `approvedSymbol`, `biotype`).
The biosample index may optionally include `obsoleteTerms`.
"""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t

_STRING_ARRAY = t.ArrayType(t.StringType())


def prepare_target_lut(df: DataFrame) -> DataFrame:
    """Explode target index rows into `(targetId, biotype, targetFromSourceId)`."""
    aliases = f.array_distinct(
        f.flatten(
            f.array(
                f.array(f.col('id')),
                f.coalesce(
                    f.transform(f.col('proteinIds'), lambda p: p.id),
                    f.array().cast(_STRING_ARRAY),
                ),
                f.array(f.col('approvedSymbol')),
            )
        )
    )
    return (
        df
        .select(
            f.col('id').alias('targetId'),
            f.col('biotype'),
            f.explode(aliases).alias('targetFromSourceId'),
        )
        .filter(f.col('targetFromSourceId').isNotNull())
        .distinct()
    )


def prepare_biosample_lut(df: DataFrame) -> DataFrame:
    """Explode biosample index rows into `(biosampleId, biosampleFromSourceMappedId)`."""
    arrays = [f.array(f.col('biosampleId'))]
    if 'obsoleteTerms' in df.columns:
        arrays.append(f.coalesce(f.col('obsoleteTerms'), f.array().cast(_STRING_ARRAY)))
    aliases = f.concat(*arrays)
    return (
        df
        .select(
            f.col('biosampleId'),
            f.explode(aliases).alias('biosampleFromSourceMappedId'),
        )
        .filter(f.col('biosampleFromSourceMappedId').isNotNull())
        .distinct()
    )
