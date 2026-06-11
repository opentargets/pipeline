"""Shared ``{label, source}`` struct primitives for drug-name datasets.

These are used by both the core ChEMBL molecule synonym wrapping
(``chembl_molecule``) and the clinical-trial synonym mining (``aact_synonyms``),
so they live in a neutral module to avoid an import cycle between the two.
"""

import pyspark.sql.functions as f
from pyspark.sql.types import ArrayType, StringType, StructField, StructType

CHEMBL_SOURCE = 'ChEMBL'
AACT_SOURCE = 'AACT'

LABEL_SOURCE_SCHEMA = ArrayType(
    StructType([
        StructField('label', StringType()),
        StructField('source', StringType()),
    ])
)


def as_label_source(label_col, source_val):
    """Wrap a string column as a ``{label, source}`` struct."""
    return f.struct(label_col.alias('label'), f.lit(source_val).alias('source'))
