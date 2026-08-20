"""Schema for the OTAR projects dataset."""

import polars as pl

project_struct_schema = pl.Struct({
    'otar_code': pl.String(),
    'status': pl.String(),
    'project_name': pl.String(),
    'integrates_data_PPP': pl.Boolean(),
    'reference': pl.String(),
})

otar_schema = pl.Schema({
    'efo_id': pl.String(),
    'projects': pl.List(project_struct_schema),
})
