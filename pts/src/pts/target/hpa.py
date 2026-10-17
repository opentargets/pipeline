"""Human Protein Atlas subcellular locations."""

from __future__ import annotations

import io
import zipfile

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.target.lists import union
from pts.transformers.utils.dataset import scan_dataset

HPA_SOURCES = {
    'Main location': 'HPA_main',
    'Additional location': 'HPA_additional',
    'Extracellular location': 'HPA_extracellular_location',
}


def read_hpa(path: str) -> pl.DataFrame:
    """The single TSV inside the HPA subcellular location zip, every column as text."""
    data, _ = StorageHandle(path).read()
    with zipfile.ZipFile(io.BytesIO(data)) as archive:
        (member,) = archive.namelist()
        content = archive.read(member)
    return pl.read_csv(
        content,
        separator='\t',
        has_header=True,
        quote_char=None,
        infer_schema=False,
        columns=['Gene', *HPA_SOURCES],
    )


def read_hpa_ssl(path: str) -> pl.DataFrame:
    """The HPA location to UniProt subcellular location term mapping."""
    return scan_dataset(path, format='tsv', has_header=True, quote_char=None, infer_schema=False).collect()


def _locations(column: str, source: str) -> pl.Expr:
    return (
        pl
        .col(column)
        .str.split(';')
        .list.eval(pl.struct(pl.element().str.strip_chars(' ').alias('location'), pl.lit(source).alias('source')))
    )


def build_hpa(hpa: pl.DataFrame, ssl: pl.DataFrame) -> pl.DataFrame:
    """HPA locations per Ensembl gene id, in file order.

    Returns:
        [id, locations], each {location, source, termSL, labelSL, targetModifier}.
    """
    return (
        hpa
        .select(
            pl.col('Gene').alias('id'),
            union(*[_locations(column, source) for column, source in HPA_SOURCES.items()]).alias('location'),
        )
        .explode('location', empty_as_null=False, keep_nulls=False)
        .unnest('location')
        .join(
            ssl.select('HPA_location', 'termSL', 'labelSL'),
            left_on='location',
            right_on='HPA_location',
            how='left',
            maintain_order='left',
        )
        .with_columns(pl.lit(None, dtype=pl.String).alias('targetModifier'))
        .group_by('id', maintain_order=True)
        .agg(pl.struct('location', 'source', 'termSL', 'labelSL', 'targetModifier').alias('locations'))
    )
