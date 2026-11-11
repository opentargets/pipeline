"""AstraZeneca PheWAS Portal ExWAS: the single-variant associations VIDRA uses as its rare-variant source.

The portal's binary ExWAS export (`azphewas-com-470k-exwas-binary.csv.bz2`, ~3.4 GB compressed) holds
one row per variant, phenotype and genetic model. VIDRA keeps the allelic model at genome-wide
significance (`GsourceLab = 1`); the kept rows are written once here, in VIDRA's column names and
with Open Targets variant identifiers, so that both the variant index (which annotates these
variants like any other source) and the `vidra_input` step read a small parquet instead of the
export.
"""

from __future__ import annotations

import bz2
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from pydantic import BaseModel, ConfigDict, Field

from pts.transformers.utils.dataset import write_dataset

COPY_BYTES = 64 << 20

COLUMNS = {
    'Variant': 'variantFromSource',
    'Variant type': 'variantType',
    'Phenotype': 'phenotype',
    'Category': 'category',
    'Model': 'model',
    'Consequence type': 'consequence',
    'Gene': 'geneSymbol',
    'Transcript': 'transcriptId',
    'No. cases': 'nCases',
    'No. controls': 'nControls',
    'Case AAF': 'caseAlleleFrequency',
    'Control AAF': 'controlAlleleFrequency',
    'p-value': 'pValue',
    'Odds ratio': 'oddsRatio',
    'Odds ratio LCI': 'oddsRatioLower',
    'Odds ratio UCI': 'oddsRatioUpper',
}
"""Export columns kept, and their names in the output."""

NUMERIC = ('nCases', 'nControls', 'caseAlleleFrequency', 'controlAlleleFrequency', 'pValue', 'oddsRatio',
           'oddsRatioLower', 'oddsRatioUpper')  # fmt: skip


class AzExwasSettings(BaseModel):
    """Which associations to keep."""

    model_config = ConfigDict(extra='forbid')

    models: list[str] = Field(default_factory=lambda: ['allelic'])
    """Genetic models kept (`Model` column); VIDRA uses the allelic test."""
    p_value_threshold: float = Field(default=5e-8, gt=0, le=1)
    """Associations with a p-value below this are kept."""


def variant_id(variant: pl.Expr) -> pl.Expr:
    """The Open Targets identifier of a portal variant: `1-100007039-G-A` becomes `1_100007039_G_A`.

    Examples:
        >>> frame = pl.DataFrame({'v': ['1-100007039-G-A', 'X-1-AT-A', '10 5 C T']})
        >>> frame.select(variant_id(pl.col('v')))['v'].to_list()
        ['1_100007039_G_A', 'X_1_AT_A', '10_5_C_T']
    """
    return variant.str.replace_all(r'[- ]', '_')


def select_associations(export: pl.LazyFrame, settings: AzExwasSettings) -> pl.LazyFrame:
    """The kept associations of the export, renamed, typed and with Open Targets variant identifiers.

    The portal quotes gene symbols and exon ranks with single quotes, which are stripped. Every
    export column is read as text; the numeric ones are cast afterwards, null when unparsable.

    Examples:
        >>> export = pl.LazyFrame({
        ...     'Variant': ['1-10-A-G', '1-10-A-G', '1-20-C-T'], 'Variant type': ['snv'] * 3,
        ...     'Phenotype': ['p'] * 3, 'Category': ['c'] * 3, 'Model': ['allelic', 'dominant', 'allelic'],
        ...     'Consequence type': ['missense_variant'] * 3, 'Gene': ["'G'"] * 3, 'Transcript': ['t'] * 3,
        ...     'No. cases': ['10'] * 3, 'No. controls': ['100'] * 3, 'Case AAF': ['0.1'] * 3,
        ...     'Control AAF': ['0.01'] * 3,
        ...     'p-value': ['1e-9', '1e-9', '0.5'], 'Odds ratio': ['3.2'] * 3, 'Odds ratio LCI': ['2'] * 3,
        ...     'Odds ratio UCI': ['5'] * 3,
        ... })
        >>> kept = select_associations(export, AzExwasSettings()).collect()
        >>> kept.select('variantId', 'geneSymbol', 'model', 'pValue', 'nCases').rows()
        [('1_10_A_G', 'G', 'allelic', 1e-09, 10)]
    """
    frame = export.rename(COLUMNS).select(list(COLUMNS.values()))
    return (
        frame
        .with_columns(
            pl.col('geneSymbol').str.strip_chars("'"),
            *[pl.col(column).cast(pl.Float64, strict=False) for column in NUMERIC],
        )
        .with_columns(pl.col('nCases', 'nControls').cast(pl.Int64, strict=False))
        .filter(pl.col('model').is_in(settings.models) & (pl.col('pValue') < settings.p_value_threshold))
        .select(variant_id(pl.col('variantFromSource')).alias('variantId'), pl.all())
    )


def _decompress(source: str, directory: Path) -> Path:
    """The export decompressed into `directory`, read from wherever `source` is."""
    target = directory / 'az_exwas.csv'
    with StorageHandle(source).open('rb') as remote, bz2.open(remote, 'rb') as stream, target.open('wb') as out:
        shutil.copyfileobj(stream, out, COPY_BYTES)
    return target


def az_exwas(source: str, destination: str, settings: dict[str, Any], config: Config) -> None:
    """Write the kept ExWAS associations of the portal export.

    Args:
        source: the `azphewas-com-470k-exwas-binary.csv.bz2` export.
        destination: parquet directory of the kept associations.
        settings: `AzExwasSettings`.
        config: for `work_path`, where the export is decompressed.
    """
    options = AzExwasSettings.model_validate(settings)
    with TemporaryDirectory(dir=config.work_path) as directory:
        logger.info('Decompressing {}', source)
        csv = _decompress(source, Path(directory))
        logger.info('Decompressed to {} ({} bytes)', csv, csv.stat().st_size)
        export = pl.scan_csv(csv, schema_overrides=dict.fromkeys(COLUMNS, pl.String), infer_schema=False)
        kept = select_associations(export, options).collect(engine='streaming')
    logger.info(
        '{} associations kept ({} variants) with models {} and p < {}',
        kept.height,
        kept['variantId'].n_unique(),
        options.models,
        options.p_value_threshold,
    )
    if kept.is_empty():
        raise ValueError('no ExWAS associations kept; check the export and the settings')
    write_dataset(kept, destination)
