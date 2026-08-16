"""OTAR projects dataset generation.

Ported from Otar.scala in platform-etl-backend.
Joins OTAR project metadata with disease EFO mappings and propagates
project info to disease ancestors.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.otar import otar_schema
from pts.transformers.utils.dataset import scan_dataset, write_dataset

REFERENCE_PREFIX = 'http://home.opentargets.org/'

# spark's cast(string as boolean) is permissive and case insensitive, and yields
# null rather than failing on anything it does not recognise. The otar metadata
# spreadsheet stores Y/N, which spark resolves to true/false; polars refuses to
# cast a string to a boolean at all, so the mapping is spelled out here.
TRUE_STRINGS = ['t', 'true', 'y', 'yes', '1']
FALSE_STRINGS = ['f', 'false', 'n', 'no', '0']


def _read_csv(path: str) -> pl.DataFrame:
    """Read a comma separated file with a header, every column as a string.

    Matches `spark.read.option('header', 'true').csv(...)` without `inferSchema`,
    which also yields all strings.
    """
    h = StorageHandle(path)
    return pl.read_csv(h.open(), has_header=True, infer_schema=False)


def _spark_string_to_boolean(column: pl.Expr) -> pl.Expr:
    """Reproduce spark's `cast(string as boolean)`."""
    normalised = column.str.strip_chars().str.to_lowercase()
    return (
        pl
        .when(normalised.is_in(TRUE_STRINGS))
        .then(pl.lit(True))
        .when(normalised.is_in(FALSE_STRINGS))
        .then(pl.lit(False))
        .otherwise(None)
    )


def _generate_otar_info(
    disease: pl.DataFrame,
    otar_meta: pl.DataFrame,
    efo_lookup: pl.DataFrame,
) -> pl.DataFrame:
    """Generate per-disease OTAR project info with ancestor propagation.

    Args:
        disease: Disease DataFrame with columns [id, ancestors].
        otar_meta: OTAR metadata with [otar_code, project_name, project_status, integrates_in_PPP].
        efo_lookup: Mapping from [otar_code, efo_disease_id].

    Returns:
        DataFrame with [efo_id, projects[{otar_code, status, project_name,
        integrates_data_PPP, reference}]].
    """
    return (
        otar_meta
        .join(efo_lookup, on='otar_code', how='left')
        .rename({'efo_disease_id': 'efo_code'})
        # inner join: projects with no mapped disease, and mappings to a disease
        # that is not in the index, both drop out here
        .join(disease.select(pl.col('id').alias('efo_code'), 'ancestors'), on='efo_code', how='inner')
        .with_columns(ancestor=pl.concat_list('efo_code', 'ancestors'))
        .explode('ancestor')
        # sorting before the group by is what fixes the element order of the
        # projects lists: the aggregation below collects rows in frame order, and
        # otar_code is unique per project
        .sort('otar_code')
        .group_by('ancestor', maintain_order=True)
        .agg(
            pl
            .struct(
                pl.col('otar_code'),
                pl.col('project_status').alias('status'),
                pl.col('project_name'),
                _spark_string_to_boolean(pl.col('integrates_in_PPP')).alias('integrates_data_PPP'),
                (pl.lit(REFERENCE_PREFIX) + pl.col('otar_code')).alias('reference'),
            )
            .unique(maintain_order=True)
            .alias('projects')
        )
        .rename({'ancestor': 'efo_id'})
        .sort('efo_id')
        .cast(otar_schema)  # type: ignore[arg-type]
    )


def otar(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate OTAR projects dataset."""
    logger.info('Reading otar inputs')
    disease = scan_dataset(source['diseases']).collect()
    meta = _read_csv(source['otar_meta'])
    lookup = _read_csv(source['otar_project_to_efo'])
    logger.info(f'{meta.height} projects, {lookup.height} project to efo mappings, {disease.height} diseases')

    result = _generate_otar_info(disease, meta, lookup)
    logger.info(f'Generated otar info for {result.height} diseases')

    logger.info(f'Writing otar output to {destination}')
    write_dataset(result, destination)
