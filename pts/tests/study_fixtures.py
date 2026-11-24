"""Typed study frames for the study deconvolution and validation tests."""

import polars as pl

from pts.schemas.study import study_schema
from pts.transformers.study import QualityFlag

# Columns the validators derive; no study input carries them.
DERIVED_COLUMNS = {'diseaseIds', 'backgroundDiseaseIds', 'biosampleId'}
source_schema = pl.Schema({name: dtype for name, dtype in study_schema.items() if name not in DERIVED_COLUMNS})


def studies_frame(rows, schema=source_schema):
    """Build typed studies, by default as the sources provide them, with absent optional fields filled with nulls."""
    rows = [
        {
            name: [str(item) if isinstance(item, QualityFlag) else item for item in value]
            if isinstance(value, list)
            else value
            for name, value in row.items()
        }
        for row in rows
    ]
    fields = {name for row in rows for name in row}
    return (
        pl
        .DataFrame(rows, schema={name: dtype for name, dtype in schema.items() if name in fields})
        .lazy()
        .match_to_schema(schema, missing_columns='insert')
    )
