"""Schemas for the coding variant step."""

import polars as pl

datasource = pl.Struct({'datasourceCount': pl.Int32(), 'datasourceId': pl.String(), 'datasourceNiceName': pl.String()})
"""One entry of the `datasources` list."""
