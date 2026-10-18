"""Inputs of the OnToma lookup table steps."""

import polars as pl

from pts.transformers.ontoma_lut.build import curation_label_lut
from pts.transformers.utils.dataset import scan_dataset
from pts.transformers.utils.spark_csv import read_spark_csv

CURATIONS = {
    'ot_disease_curation': 'ot_curation',
    'eva_clinvar': 'eva_clinvar',
    'clinvar_xrefs': 'clinvar_xrefs',
}


def read_parquet(path: str) -> pl.DataFrame:
    """Read a parquet dataset."""
    return scan_dataset(path).collect()


def curation_label_luts(source: dict[str, str], disease: pl.DataFrame) -> list[pl.DataFrame]:
    """The label lookup tables of the three disease curations."""
    return [
        curation_label_lut(read_spark_csv(source[key], separator='\t'), name, disease)
        for key, name in CURATIONS.items()
    ]
