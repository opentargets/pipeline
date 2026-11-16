"""Baseline expression: merge the aggregated datasets, resolve their targets and biosamples, and flag what fails.

Each dataset under `<baseline_expression>/<aggregation>/<dataset>/parquet` is one source's
expression summary per target and biosample. The datasets are concatenated, PRIDE's UniProt
accessions are resolved to Ensembl genes, and every tissue and cell type gets a parent biosample
from a curated list of candidate parents. Targets and biosamples are then checked against the
target and biosample indexes; rows that fail either check go to `failed`.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.transformers.utils.dataset import scan_dataset, scan_datasets, sink_dataset
from pts.transformers.utils.quality_flags import QC_COLUMN, flag_rows
from pts.transformers.utils.spark_csv import read_spark_csv
from pts.transformers.utils.spark_strings import spark_trim

PRIDE = 'PRIDE'

#: The curated parents' key column, and the column that is not a label.
PARENT_KEY = 'BiosampleId'
PARENT_COUNT = 'Count'

#: The published column order; columns a dataset adds beyond these follow them.
COLUMN_ORDER = [
    'celltypeBiosampleId',
    'tissueBiosampleId',
    'targetId',
    'datasourceId',
    'datatypeId',
    'unit',
    'celltypeBiosampleFromSource',
    'min',
    'q1',
    'median',
    'q3',
    'max',
    'distribution_score',
    'specificity_score',
    'tissueBiosampleFromSource',
    'targetFromSourceId',
    'tissueBiosampleParentId',
    'celltypeBiosampleParentId',
    QC_COLUMN,
]

#: For each biosample dimension: its id column, the parent column derived from it, and the source label.
DIMENSIONS = {
    'tissue': ('tissueBiosampleId', 'tissueBiosampleParentId', 'tissueBiosampleFromSource'),
    'celltype': ('celltypeBiosampleId', 'celltypeBiosampleParentId', 'celltypeBiosampleFromSource'),
}

#: Java's `\s`, which is ASCII-only where Rust's is unicode-aware.
_JAVA_SPACE = r'[ \t\n\x0B\f\r]+'


class BaselineExpressionFlags(StrEnum):
    """Reasons a baseline expression row is excluded."""

    INVALID_TARGET = 'No valid target'
    INVALID_BIOSAMPLE = 'No valid biosample'


def normalise_label(label: pl.Expr) -> pl.Expr:
    """A label reduced to its lowercase ASCII letters and digits, for matching labels written differently."""
    return label.str.to_lowercase().str.replace_all(r'[^a-z0-9]', '')


def prettify_label(label: pl.Expr) -> pl.Expr:
    """A label split into lowercase words, keeping acronyms and single capitals as written.

    `AdiposeVisceralOmentum` becomes `adipose visceral omentum` and `BrainFrontalCortexBA9`
    becomes `brain frontal cortex BA9`.
    """
    spaced = (
        label
        .fill_null('')
        .str.replace_all(r'[_-]+', ' ')
        .str.replace_all(r'([A-Z])([A-Z][a-z])', '$1 $2')
        .str.replace_all(r'([a-z])([A-Z])', '$1 $2')
        .str.replace_all(r'([a-z])([0-9])', '$1 $2')
        .str.replace_all(r'([0-9])([a-z])', '$1 $2')
        .str.replace_all(r'[.,]', '')
        .str.replace_all(_JAVA_SPACE, ' ')
    )
    words = spark_trim(spaced).str.split(' ')
    kept = pl.element().str.contains(r'^[A-Z0-9]{2,}$') | pl.element().str.contains(r'^[A-Z]$')
    return words.list.eval(pl.when(kept).then(pl.element()).otherwise(pl.element().str.to_lowercase())).list.join(' ')


def target_aliases(target: pl.LazyFrame) -> pl.DataFrame:
    """Each target id, protein id and approved symbol, with the one target it names.

    An alias naming several targets resolves to the smallest target id.
    """
    proteins = target.select(
        pl.col('id').alias('targetId'),
        pl.col('proteinIds').list.eval(pl.element().struct.field('id')).alias('alias'),
    ).explode('alias')
    aliases = pl.concat([
        target.select(pl.col('id').alias('targetId'), pl.col('id').alias('alias')),
        proteins,
        target.select(pl.col('id').alias('targetId'), pl.col('approvedSymbol').alias('alias')),
    ])
    return aliases.drop_nulls('alias').group_by('alias').agg(pl.col('targetId').min()).collect()


def biosample_aliases(biosample: pl.DataFrame) -> pl.DataFrame:
    """Each biosample id and obsolete term, with the one biosample it names (the smallest id where several)."""
    frames = [biosample.select('biosampleId', pl.col('biosampleId').alias('alias'))]
    if 'obsoleteTerms' in biosample.columns:
        frames.append(biosample.select('biosampleId', pl.col('obsoleteTerms').alias('alias')).explode('alias'))
    return pl.concat(frames).drop_nulls('alias').group_by('alias').agg(pl.col('biosampleId').min())


def resolve_pride_targets(expression: pl.LazyFrame, target: pl.LazyFrame) -> pl.LazyFrame:
    """Replace PRIDE's target with the genes its UniProt accession maps to, or the accession where none does.

    An accession mapping to several genes gives one row per gene.
    """
    if 'targetFromSourceId' not in expression.collect_schema():
        logger.warning('no targetFromSourceId column; skipping PRIDE target resolution')
        return expression
    genes = (
        target
        .select(
            pl.col('id').alias('_resolvedTargetId'),
            pl.col('proteinIds').list.eval(pl.element().struct.field('id')).alias('_proteinId'),
        )
        .explode('_proteinId')
        .drop_nulls('_proteinId')
        .unique()
    )
    is_pride = pl.col('datasourceId') == PRIDE
    return (
        expression
        .with_columns(pl.when(is_pride).then(pl.col('targetFromSourceId')).alias('_proteinId'))
        .join(genes, on='_proteinId', how='left')
        .with_columns(
            pl
            .when(is_pride)
            .then(pl.coalesce('_resolvedTargetId', 'targetFromSourceId'))
            .otherwise(pl.col('targetId'))
            .alias('targetId')
        )
        .drop('_proteinId', '_resolvedTargetId')
    )


def parent_by_biosample(biosample: pl.DataFrame, candidates: list[str]) -> pl.DataFrame:
    """For each biosample, the most specific candidate parent that is the biosample itself or one of its ancestors.

    A candidate with more ancestors is more specific; candidates tied on that keep their listed order,
    and a candidate missing from the index ranks last.
    """
    ancestor_counts = dict(
        biosample
        .filter(pl.col('biosampleId').is_in(candidates))
        .select('biosampleId', pl.col('ancestors').list.len().fill_null(0))
        .iter_rows()
    )
    ranked = sorted(candidates, key=lambda candidate: ancestor_counts.get(candidate, -1), reverse=True)
    priority = pl.DataFrame(
        {'_candidate': ranked, '_priority': range(len(ranked))},
        schema={'_candidate': pl.String, '_priority': pl.Int64},
    )
    return (
        biosample
        .select(
            'biosampleId',
            pl.concat_list(pl.col('biosampleId'), pl.col('ancestors').fill_null([])).list.unique().alias('_candidate'),
        )
        .explode('_candidate')
        .join(priority, on='_candidate', how='inner')
        .group_by('biosampleId')
        .agg(pl.col('_candidate').sort_by('_priority').first().alias('parentId'))
    )


def parent_by_label(parents: pl.DataFrame) -> pl.DataFrame:
    """The curated parents keyed by the normalised form of every label they list.

    Every column but the key and the count holds `;`-separated labels. A label listed under
    several parents goes to the smallest parent id, and its prettified form replaces the
    source label of a row it matches. Where one parent lists a label in several spellings,
    the spelling in the leftmost column is used.
    """
    labels = [column for column in parents.columns if column not in (PARENT_KEY, PARENT_COUNT)]
    long = (
        parents
        .unpivot(on=labels, index=PARENT_KEY, variable_name='_column', value_name='_label')
        .with_columns(
            pl.col('_column').replace_strict(labels, range(len(labels)), return_dtype=pl.Int64),
            pl.col('_label').fill_null('').str.split(';'),
        )
        .explode('_label')
        .with_columns(spark_trim(pl.col('_label')))
        .filter(pl.col('_label') != '')
        .with_columns(
            normalise_label(pl.col('_label')).alias('_norm'),
            prettify_label(pl.col('_label')).alias('_pretty'),
        )
        .filter(pl.col('_norm') != '')
    )
    return (
        long
        .with_row_index('_position')
        .sort(PARENT_KEY, '_column', '_position')
        .group_by('_norm', maintain_order=True)
        .agg(pl.col(PARENT_KEY).first().alias('_labelParentId'), pl.col('_pretty').first())
    )


def add_parent(
    expression: pl.LazyFrame,
    biosample: pl.DataFrame,
    parents: pl.DataFrame,
    dimension: str,
) -> pl.LazyFrame:
    """Add a dimension's parent biosample, from the biosample's ancestry or else from its source label.

    A row whose parent comes from its source label has that label replaced by the curated form.
    """
    id_column, parent_column, source_column = DIMENSIONS[dimension]
    schema = expression.collect_schema()
    if id_column not in schema or source_column not in schema:
        logger.warning(f'no {id_column} or {source_column} column; skipping {parent_column}')
        return expression
    by_ancestry = parent_by_biosample(biosample, parents.get_column(PARENT_KEY).to_list()).rename({
        'biosampleId': id_column,
        'parentId': parent_column,
    })
    by_label = parent_by_label(parents)
    return (
        expression
        .join(by_ancestry.lazy(), on=id_column, how='left')
        .with_columns(
            pl.when(pl.col(parent_column).is_null()).then(normalise_label(pl.col(source_column))).alias('_norm')
        )
        .join(by_label.lazy(), on='_norm', how='left')
        .with_columns(
            pl.coalesce(parent_column, '_labelParentId').alias(parent_column),
            pl.coalesce('_pretty', source_column).alias(source_column),
        )
        .drop('_norm', '_labelParentId', '_pretty')
    )


def validate_target(expression: pl.LazyFrame, aliases: pl.DataFrame) -> pl.LazyFrame:
    """Resolve `targetId` through the target aliases, flagging rows it names no target."""
    lut = aliases.lazy().select(pl.col('alias').alias('targetId'), pl.col('targetId').alias('_mapped'))
    return (
        expression
        .join(lut, on='targetId', how='left')
        .pipe(flag_rows, pl.col('_mapped').is_null(), BaselineExpressionFlags.INVALID_TARGET)
        .with_columns(pl.coalesce('_mapped', 'targetId').alias('targetId'))
        .drop('_mapped')
    )


def validate_biosample(
    expression: pl.LazyFrame,
    aliases: pl.DataFrame,
    id_column: str,
    source_column: str,
) -> pl.LazyFrame:
    """Resolve `id_column` through the biosample aliases, flagging rows with a biosample it names none of.

    A row has a biosample in this dimension when it has an id or a source label for it.
    """
    schema = expression.collect_schema()
    if id_column not in schema and source_column not in schema:
        return expression
    if id_column not in schema:
        expression = expression.with_columns(pl.lit(None, dtype=pl.String).alias(id_column))
    required = pl.col(id_column).is_not_null()
    if source_column in schema:
        required = required | pl.col(source_column).is_not_null()
    lut = aliases.lazy().select(pl.col('alias').alias(id_column), pl.col('biosampleId').alias('_mapped'))
    return (
        expression
        .join(lut, on=id_column, how='left')
        .pipe(flag_rows, required & pl.col('_mapped').is_null(), BaselineExpressionFlags.INVALID_BIOSAMPLE)
        .with_columns(pl.coalesce('_mapped', id_column).alias(id_column))
        .drop('_mapped')
    )


def order_columns(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Put the published columns first, in their published order."""
    names = frame.collect_schema().names()
    return frame.select(
        [column for column in COLUMN_ORDER if column in names] + [c for c in names if c not in COLUMN_ORDER]
    )


def build(
    expression: pl.LazyFrame,
    target: pl.LazyFrame,
    biosample: pl.DataFrame,
    tissue_parents: pl.DataFrame,
    celltype_parents: pl.DataFrame,
) -> pl.LazyFrame:
    """Resolve targets and parents, then flag every row failing validation."""
    resolved = (
        expression
        .pipe(resolve_pride_targets, target)
        .pipe(add_parent, biosample, tissue_parents, 'tissue')
        .pipe(add_parent, biosample, celltype_parents, 'celltype')
    )
    biosamples = biosample_aliases(biosample)
    validated = validate_target(resolved, target_aliases(target))
    for id_column, parent_column, source_column in DIMENSIONS.values():
        validated = validate_biosample(validated, biosamples, id_column, source_column)
        validated = validate_biosample(validated, biosamples, parent_column, source_column)
    return order_columns(validated)


def _read_parents(path: str) -> pl.DataFrame:
    return read_spark_csv(path, separator='\t').drop(PARENT_COUNT, strict=False)


def baseline_expression(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    _config: Config | None = None,
) -> None:
    """Merge, resolve and validate the aggregated baseline expression datasets.

    Args:
        source: `baseline_expression` (the root of the aggregated datasets), `target_index`,
            `biosample_index`, `tissue_parents` and `celltype_parents`.
        destination: `valid` and `failed`.
        settings: `aggregation`, the aggregation level directory, and `datasets`, each dataset's
            path below it.
        _config: unused.
    """
    root = source['baseline_expression'].rstrip('/')
    paths = [f'{root}/{settings["aggregation"]}/{dataset.strip("/")}/parquet' for dataset in settings['datasets']]
    for path in paths:
        logger.info(f'reading {path}')

    target = scan_dataset(source['target_index']).select('id', 'proteinIds', 'approvedSymbol')
    biosample = scan_dataset(source['biosample_index']).collect()
    flagged = build(
        scan_datasets(paths),
        target,
        biosample,
        _read_parents(source['tissue_parents']),
        _read_parents(source['celltype_parents']),
    )
    valid = flagged.filter(pl.col(QC_COLUMN).list.len() == 0)
    failed = flagged.filter(pl.col(QC_COLUMN).list.len() != 0)
    logger.info(f'writing {destination["valid"]} and {destination["failed"]}')
    pl.collect_all([sink_dataset(valid, destination['valid']), sink_dataset(failed, destination['failed'])])
