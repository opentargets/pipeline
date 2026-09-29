"""Look-up tables used to validate evidence against the disease, target and literature indices.

Polars port of `pts.pyspark.evidence_utils.validation_lut.LookUpTables`. Only the fields evidence
validation needs are ported -- `TSorOncogene` (cancer-hallmark gene assessment, used only for
`assign_direction_on_target`) is out of scope, since no evidence source migrated so far sets a
`direction_on_target_expression`.
"""

import polars as pl

_LITERATURE_SOURCES = ('MED', 'PPR', 'AGR')


def build_disease_lut(disease_df: pl.DataFrame) -> pl.DataFrame:
    """Build a disease look-up table mapping source disease IDs (current and obsolete) to `diseaseId`.

    Args:
        disease_df: raw disease index with `id` and `obsoleteTerms` (nullable list[str]) columns.

    Returns:
        DataFrame with `diseaseId` and `diseaseFromSourceMappedId` -- one row per `id`, plus one
        row per obsolete term that used to identify it.
    """
    return (
        disease_df
        .select(
            pl.col('id').alias('diseaseId'),
            pl.concat_list([pl.col('id'), pl.col('obsoleteTerms').fill_null([])]).alias('diseaseFromSourceMappedId'),
        )
        .explode('diseaseFromSourceMappedId', empty_as_null=True)
        .unique()
    )


def build_target_lut(target_df: pl.DataFrame) -> pl.DataFrame:
    """Build a target look-up table mapping source target IDs to `targetId` and `biotype`.

    Args:
        target_df: raw target index with `id`, `biotype`, `proteinIds` (nullable
            list[struct{id: str, ...}]) and `approvedSymbol` columns.

    Returns:
        DataFrame with `targetId`, `biotype` and `targetFromSourceId` -- one row per Ensembl gene
        ID, UniProt protein ID and approved symbol that can identify the target.
    """
    return (
        target_df
        .select(
            pl.col('id').alias('targetId'),
            pl.col('biotype'),
            pl.concat_list([
                pl.col('id'),
                pl.col('proteinIds').list.eval(pl.element().struct.field('id')).fill_null([]),
                pl.col('approvedSymbol'),
            ]).alias('targetFromSourceIds'),
        )
        .explode('targetFromSourceIds', empty_as_null=True)
        .rename({'targetFromSourceIds': 'targetFromSourceId'})
        .drop_nulls('targetFromSourceId')
        .unique()
    )


def build_publication_lut(literature_df: pl.DataFrame) -> pl.DataFrame:
    """Build a publication look-up table mapping publication IDs to their first publication date.

    Args:
        literature_df: raw literature export with `source`, `firstPublicationDate`, `pmid`, `id`
            and `pmcid` columns.

    Returns:
        DataFrame with `publicationDate` and `publicationId` -- one row per non-null identifier
        (`pmid`, `id` or `pmcid`) for publications from MED, PPR or AGR sources.
    """
    return (
        literature_df
        .filter(pl.col('source').is_in(_LITERATURE_SOURCES))
        .select(
            pl.col('firstPublicationDate').alias('publicationDate'),
            pl.concat_list([pl.col('pmid'), pl.col('id'), pl.col('pmcid')]).alias('publicationId'),
        )
        .explode('publicationId', empty_as_null=True)
        .drop_nulls('publicationId')
    )
