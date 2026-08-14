"""Look up tables used to validate evidence.

Polars port of `pts.pyspark.evidence_utils.validation_lut.LookUpTables`. Every builder
below was diffed against the spark implementation on the real 26.06 release: disease
54,961 rows, target 511,837 rows and the publication table over three of the export's
56 parts, all exact matches including `TSorOncogene`.

Divergences worth naming, because each one is deliberate:

* The disease table is **not** deduplicated. Spark does not deduplicate it either, and
  10 diseases in the 26.06 index list an identifier twice, so the duplicate rows fan
  the matching evidence out. `Evidence.validate_uniqueness` is what later flags the
  extra copy, which puts it in `failed_evidence` rather than dropping it.
* The publication table **is** deduplicated, where spark does not. Its only consumer
  keeps the earliest date per evidence and then deduplicates the result, so removing
  rows that are identical in both columns cannot change the answer, and 44% of the
  real rows are exactly that: `pmid` and `id` hold the same value for most MED records.
* `pl.concat_list` returns null for the whole row when any argument is a null list,
  so the list arguments are filled first. This mirrors spark, whose `concat` returns
  null the same way and which the spark implementation guards with `coalesce`.
"""

from __future__ import annotations

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

LITERATURE_SOURCES = ['MED', 'PPR', 'AGR']

# Polars infers a newline delimited json column's dtype from a sample of the leading
# rows. `pmid` and `pmcid` are absent from the first rows of some parts of the real
# literature export, which infers them as Null and then either fails the read or, under
# `ignore_errors`, silently discards every value in the column — 386,627 pmids in one of
# the 26.06 export's 56 parts. Pinning the columns the table needs makes the read
# independent of where the nulls happen to fall, and lets the parts be read separately.
LITERATURE_SCHEMA = {
    'source': pl.String,
    'pmid': pl.String,
    'id': pl.String,
    'pmcid': pl.String,
    'firstPublicationDate': pl.String,
}


def _parts(path: str, pattern: str) -> list[str]:
    """List the part files of a dataset directory.

    Args:
        path: location of the dataset directory.
        pattern: glob matching its part files, which also excludes markers like `_SUCCESS`.

    Returns:
        The part locations, sorted so a run does not depend on listing order.

    Raises:
        ValueError: if the directory holds no matching part, which otherwise reads as
            an empty dataset and only surfaces much later as unmapped evidence.
    """
    parts = sorted(StorageHandle(path).glob(pattern))
    if not parts:
        raise ValueError(f'no {pattern} files found in {path}')
    return parts


def _read_parquet(path: str) -> pl.DataFrame:
    """Read a parquet dataset directory through the storage backend."""
    return pl.concat([pl.read_parquet(StorageHandle(part).open()) for part in _parts(path, '*.parquet')])


def _cancer_gene_assessment() -> pl.Expr:
    """Flag a gene as oncogene, tsg or bivalent from its cancer hallmark attributes.

    Cancer hallmark annotation provides a list of assessments of the gene. If any of them
    suggests an oncogenic role the gene is flagged `oncogene`, if any suggests a tumour
    suppressor function it is flagged `tsg`, and a gene carrying both is flagged
    `bivalent` — whether the two roles come from one assessment or from two.

    Returns:
        Expression yielding the assessment, or null when there is no hallmark evidence
        either way, which is the case for 78,341 of the 78,691 real targets.
    """
    descriptions = (
        pl.col('hallmarks')
        .struct.field('attributes')
        .list.eval(pl.element().struct.field('description').str.to_lowercase())
    )
    # `literal=True` because spark's `Column.contains` matches a substring, not a regex.
    has_oncogene = descriptions.list.eval(pl.element().str.contains('oncogene', literal=True)).list.any()
    has_tsg = descriptions.list.eval(pl.element().str.contains('tsg', literal=True)).list.any()
    return (
        pl.when(has_oncogene & has_tsg)
        .then(pl.lit('bivalent'))
        .when(has_oncogene)
        .then(pl.lit('oncogene'))
        .when(has_tsg)
        .then(pl.lit('tsg'))
        .otherwise(None)
    )


def build_disease_lut(path: str) -> pl.DataFrame:
    """Map every disease identifier, current or obsolete, onto its current id.

    Args:
        path: location of the disease index.

    Returns:
        DataFrame with columns `diseaseId` and `diseaseFromSourceMappedId`. Rows are
        deliberately left duplicated; see the module docstring.
    """
    return (
        _read_parquet(path)
        .select(
            pl.col('id').alias('diseaseId'),
            pl.concat_list(pl.col('id'), pl.col('obsoleteTerms').fill_null([])).alias('diseaseFromSourceMappedId'),
        )
        .explode('diseaseFromSourceMappedId')
    )


def build_target_lut(path: str) -> pl.DataFrame:
    """Map every target identifier a source might use onto the canonical target id.

    Args:
        path: location of the target index.

    Returns:
        DataFrame with columns `targetId`, `biotype`, `TSorOncogene` and
        `targetFromSourceId`.
    """
    return (
        _read_parquet(path)
        .select(
            pl.col('id').alias('targetId'),
            'biotype',
            _cancer_gene_assessment().alias('TSorOncogene'),
            pl.concat_list(
                pl.col('id'),
                pl.col('proteinIds').list.eval(pl.element().struct.field('id')).fill_null([]),
                pl.col('approvedSymbol'),
            )
            # `maintain_order` because spark's `array_distinct` keeps first occurrences
            .list.unique(maintain_order=True)
            .alias('targetFromSourceId'),
        )
        .explode('targetFromSourceId')
        .unique()
    )


def _publication_part(part: str) -> pl.DataFrame:
    """Read and reduce one part of the literature export.

    The projection runs per part rather than after concatenating, because the export is
    53.7M rows and only the two output columns survive it.
    """
    return (
        pl.read_ndjson(StorageHandle(part).open(), schema_overrides=LITERATURE_SCHEMA)
        .filter(pl.col('source').is_in(LITERATURE_SOURCES))
        .select(
            pl.col('firstPublicationDate').alias('publicationDate'),
            pl.concat_list(
                pl.col('pmid').cast(pl.String),
                pl.col('id').cast(pl.String),
                pl.col('pmcid').cast(pl.String),
            ).alias('publicationId'),
        )
        .explode('publicationId')
        .drop_nulls('publicationId')
    )


def build_publication_lut(path: str) -> pl.DataFrame:
    """Map every publication identifier onto its publication date.

    Args:
        path: location of the literature export.

    Returns:
        DataFrame with columns `publicationDate` and `publicationId`.
    """
    return pl.concat([_publication_part(part) for part in _parts(path, '*.json*')]).unique()
