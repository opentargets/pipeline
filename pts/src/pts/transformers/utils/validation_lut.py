"""Look up tables used to validate evidence.

Polars port of `pts.pyspark.evidence_utils.validation_lut.LookUpTables`. Every builder
below was diffed against the spark implementation on the real 26.06 release: disease
54,961 rows, target 511,837 rows and the publication table over three of the export's
56 parts, all exact matches including `TSorOncogene`. `build_publication_lut` was
additionally run end to end over the full 56-part export (53,703,675 rows) to confirm
the read itself completes -- see its `schema=` note for the defect that surfaced.

This module holds no reading logic of its own: every builder goes through
`pts.transformers.utils.dataset.scan_dataset`.

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

from pts.schemas.literature import literature_schema
from pts.transformers.utils.dataset import scan_dataset

LITERATURE_SOURCES = ['MED', 'PPR', 'AGR']


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
        scan_dataset(path)
        .collect()
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
        scan_dataset(path)
        .collect()
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


def build_publication_lut(path: str) -> pl.DataFrame:
    """Map every publication identifier onto its publication date.

    One lazy scan over every part, collected through the STREAMING engine. This replaced a python
    loop that read each part eagerly and projected it down to two columns before concatenating --
    a hand-rolled version of the projection pushdown the query optimiser now does, written before
    this module had a lazy reader.

    Measured on the real 56-part, 404 MB export (53,703,675 rows out, byte-identical between all
    three):

        per-part eager loop     15.4s   6.45 GiB peak
        one lazy scan, default  12.3s   9.13 GiB peak
        one lazy scan, STREAMING 4.9s   7.37 GiB peak

    So `engine='streaming'` is the point of this, not incidental: the default engine is both
    slower and hungrier than the loop it replaces, while streaming is 3.1x faster than the loop
    for 0.9 GiB more. If a future polars makes streaming the default, this argument can go.

    `schema=` (not `schema_overrides=`) is load-bearing and comes from `pts.schemas.literature`:
    overrides pins only the named columns and still infers every other column from its leading
    rows, and one real 26.06 part has a `dateOfPublication` that infers Null there and later holds
    a non-null value, raising on a column this table never even selects.

    Args:
        path: location of the literature export.

    Returns:
        DataFrame with columns `publicationDate` and `publicationId`.
    """
    return (
        scan_dataset(path, format='ndjson', schema=literature_schema)
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
        .unique()
        .collect(engine='streaming')
    )
