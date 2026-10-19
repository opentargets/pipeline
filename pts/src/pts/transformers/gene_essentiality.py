"""Gene essentiality dataset generation.

Parses the DepMap CRISPR gene effect, expression and mutation matrices into per-tissue screens, then
resolves gene symbols to ENSG ids against the target dataset. Symbols with no match are dropped.
"""

from __future__ import annotations

import csv
import io
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.target import ensg_lookup

NON_GENE_COLUMNS = {'depmapId', 'SequencingID', 'ModelConditionID', 'IsDefaultEntryForModel', 'IsDefaultEntryForMC'}
SCREEN_FIELDS = [
    'depmapId',
    'cellLineName',
    'diseaseFromSource',
    'diseaseCellLineId',
    'mutation',
    'geneEffect',
    'expression',
]
BLANK = ''.join(map(chr, range(33)))


def gene_symbol(column: str) -> pl.Expr:
    """The gene symbol of a DepMap gene label such as `AAMP (14)`."""
    return pl.col(column).str.split(' ').list.first()


def to_float32(column: str) -> pl.Expr:
    """Parse a numeric string as spark's CSV reader does, then narrow it to Float32."""
    text = pl.col(column).str.strip_chars(BLANK).str.replace(r'([0-9.])[dDfF]$', '${1}')
    return text.cast(pl.Float64, strict=False).cast(pl.Float32)


def scan_strings(path: str) -> pl.LazyFrame:
    """Lazily read a DepMap CSV with every column as a string, taking the column names from its header line."""
    with StorageHandle(path).open('rb') as f:
        header = next(csv.reader(io.TextIOWrapper(f, encoding='utf-8')))
    return scan_dataset(path, format='csv', schema=dict.fromkeys(header, pl.String), null_values=[''])


def read_models(models: pl.LazyFrame) -> pl.LazyFrame:
    """Select the cell line fields of the DepMap models.

    Args:
        models: the DepMap `Model.csv`, all columns as strings.

    Returns:
        Columns `depmapId`, `cellLineName`, `modelId`, `tissueFromSource` and `diseaseFromSource`.
    """
    return models.select(
        pl.col('ModelID').alias('depmapId'),
        pl.coalesce(
            'CellLineName',
            'CCLEName',
            pl.concat_str(pl.col('OncotreePrimaryDisease'), pl.lit(' cells')),
        ).alias('cellLineName'),
        pl.col('SangerModelID').alias('modelId'),
        pl.col('OncotreeLineage').str.to_lowercase().alias('tissueFromSource'),
        pl.col('OncotreePrimaryDisease').alias('diseaseFromSource'),
    )


def melt_matrix(lf: pl.LazyFrame, value_name: str) -> pl.LazyFrame:
    """Unpivot a DepMap model by gene matrix into one row per model and gene.

    The model id is the `ModelID` column when there is one, else the unnamed first column.

    Args:
        lf: the matrix, all columns as strings.
        value_name: name of the value column.

    Returns:
        Columns `depmapId`, `targetSymbol` and `value_name` as Float32.
    """
    columns = lf.collect_schema().names()
    if 'ModelID' in columns:
        lf = lf.drop('', strict=False).rename({'ModelID': 'depmapId'})
    elif '' in columns:
        lf = lf.rename({'': 'depmapId'})
    genes = [c for c in lf.collect_schema().names() if c not in NON_GENE_COLUMNS]
    return lf.unpivot(on=genes, index='depmapId', variable_name='geneLabel', value_name=value_name).select(
        'depmapId',
        gene_symbol('geneLabel').alias('targetSymbol'),
        to_float32(value_name).alias(value_name),
    )


def build_screens(
    models: pl.LazyFrame,
    essential_genes: pl.LazyFrame,
    gene_effects: pl.LazyFrame,
    gene_expression: pl.LazyFrame,
    mutation_hotspots: pl.LazyFrame,
    mutation_damaging: pl.LazyFrame,
    depmap_tissue_mapping: pl.LazyFrame,
    keep_only_essentials: bool = True,
) -> pl.LazyFrame:
    """Join the DepMap inputs into one row per gene and screen.

    Args:
        models: the DepMap models.
        essential_genes: the DepMap common essential genes.
        gene_effects: the CRISPR gene effect matrix.
        gene_expression: the expression matrix.
        mutation_hotspots: the hotspot mutation matrix.
        mutation_damaging: the damaging mutation matrix.
        depmap_tissue_mapping: lineage to tissue mapping.
        keep_only_essentials: keep only the common essential genes.

    Returns:
        One row per gene and screen with a gene effect.
    """
    essentials = essential_genes.select(
        gene_symbol('Essentials').alias('targetSymbol'),
        pl.lit(True).alias('isEssential'),
    )
    hotspots = melt_matrix(mutation_hotspots, 'hotspotMutation').filter(pl.col('hotspotMutation') > 0)
    damaging = melt_matrix(mutation_damaging, 'damagingMutation').filter(pl.col('damagingMutation') > 0)
    screens = (
        melt_matrix(gene_effects, 'geneEffect')
        .join(essentials, on='targetSymbol', how='left')
        .join(read_models(models), on='depmapId', how='left')
        .join(melt_matrix(gene_expression, 'expression'), on=['targetSymbol', 'depmapId'], how='left')
        .join(damaging, on=['targetSymbol', 'depmapId'], how='left')
        .join(hotspots, on=['targetSymbol', 'depmapId'], how='left')
        .join(depmap_tissue_mapping, on='tissueFromSource', how='left')
        .select(
            'targetSymbol',
            'depmapId',
            'cellLineName',
            pl.col('modelId').alias('diseaseCellLineId'),
            'diseaseFromSource',
            'tissueId',
            pl.col('tissueName').fill_null('other'),
            pl
            .when(pl.col('damagingMutation').is_not_null())
            .then(pl.lit('damaging'))
            .when(pl.col('hotspotMutation').is_not_null())
            .then(pl.lit('hotspot'))
            .alias('mutation'),
            'geneEffect',
            'expression',
            pl.col('isEssential').fill_null(False),
        )
        .filter(pl.col('geneEffect').is_not_null())
    )
    if keep_only_essentials:
        screens = screens.filter(pl.col('isEssential'))
    return screens


def aggregate(screens: pl.LazyFrame) -> pl.LazyFrame:
    """Nest the distinct screens by tissue, then the tissues by gene.

    Args:
        screens: the output of `build_screens`.

    Returns:
        Columns `targetSymbol`, `isEssential` and `depMapEssentiality`, the tissues sorted by
        id and name, and each tissue's screens sorted by every field.
    """
    return (
        screens
        .unique()
        .group_by('targetSymbol', 'isEssential', 'tissueId', 'tissueName')
        .agg(pl.struct(SCREEN_FIELDS).sort_by(SCREEN_FIELDS).alias('screens'))
        .group_by('targetSymbol', 'isEssential')
        .agg(
            pl
            .struct('tissueId', 'tissueName', 'screens')
            .sort_by('tissueId', 'tissueName')
            .alias('depMapEssentiality')
        )
    )


def resolve_target_ids(essentiality: pl.LazyFrame, lookup: pl.LazyFrame) -> pl.LazyFrame:
    """Resolve gene symbols to ENSG ids, merging the symbols that resolve to the same id.

    Args:
        essentiality: the output of `aggregate`.
        lookup: the output of `ensg_lookup`.

    Returns:
        Columns `targetId`, `isEssential` and `depMapEssentiality`, one row per ENSG id.
    """
    return (
        essentiality
        .join(lookup, left_on='targetSymbol', right_on='name', how='inner')
        .group_by(pl.col('ensgId').alias('targetId'))
        .agg(
            pl.col('isEssential').max(),
            pl.col('depMapEssentiality').sort_by('targetSymbol', 'isEssential').explode(),
        )
        .with_columns(pl.col('depMapEssentiality').list.unique(maintain_order=True))
    )


def gene_essentiality(
    source: dict[str, str],
    destination: str,
    settings: dict[str, Any] | None,
    config: Config | None,
) -> None:
    """Build the gene essentiality dataset.

    Args:
        source: paths of the DepMap CSVs (`models`, `essential_genes`, `gene_effects`,
            `gene_expression`, `mutation_hotspots`, `mutation_damaging`), the
            `depmap_tissue_mapping` CSV and the `target` dataset.
        destination: path of the gene essentiality dataset.
        settings: `keep_only_essentials`, default True.
        config: the otter config, unused.
    """
    keep_only_essentials = (settings or {}).get('keep_only_essentials', True)
    logger.debug(f'loading data from: {source}')
    frames = {name: scan_strings(path) for name, path in source.items() if name != 'target'}
    target = scan_dataset(source['target']).select('id', 'approvedSymbol', 'proteinIds')
    screens = build_screens(**frames, keep_only_essentials=keep_only_essentials)
    result = resolve_target_ids(aggregate(screens), ensg_lookup(target)).sort('targetId')
    logger.info(f'writing gene essentiality to {destination}')
    write_dataset(result, destination)
