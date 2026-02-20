"""Module to generate view for Open Target's VEP plugin."""

from pathlib import Path
from typing import Any

from loguru import logger
from pyspark.sql import Column, DataFrame
from pyspark.sql import functions as f
from pyspark.sql import types as t
from pyspark.sql.window import Window

from pts.pyspark.common import Session


def process_biosample(biosample_index: DataFrame) -> DataFrame:
    return biosample_index.select(
        f.col('biosampleId').alias('qtlBiosampleId'), f.col('biosampleName').alias('qtlBiosampleName')
    ).distinct()


def process_credible_set(credible_set: DataFrame) -> DataFrame:
    """Flatten a credible set into one row per tag variant and compute lead-variant flag.

    This function explodes the `locus` array of struct entries from a credible-set
    DataFrame so each tag variant becomes its own row. It preserves locus/study
    identifiers and key finemapping fields, and computes `isLead` by comparing the
    top-level `variantId` column to the exploded `tagVariant.variantId`.

    Notes:
      - Rows are created by exploding `locus`; if `locus` is empty/null the row will be dropped by the explode.
      - The function only computes a simple equality-based `isLead` flag; if your upstream
        data uses different identifiers (e.g., including allele), normalise them before calling.
      - Use `f.col('...')` or `f.expr(...)` when composing additional expressions to avoid SQL parsing pitfalls.

    Expected input columns (at minimum):
      - `studyLocusId` (string): locus identifier for the credible set
      - `studyId` (string): study identifier
      - `variantId` (string): canonical/lead variant id for the credible set (used to compute `isLead`)
      - `locus` (array<struct>): array of tag-variant structs containing:
          - `pValueMantissa`
          - `pValueExponent`
          - `beta`
          - `is95CredibleSet`
          - `is99CredibleSet`
          - `posteriorProbability`
          - `variantId`
      - `finemappingMethod` (string): method used to generate the credible set

    Args:
        credible_set (DataFrame): gentropy credible set

    Returns:
        DataFrame with columns:
        - `studyLocusId`, `studyId`
        - `pValueMantissa`, `pValueExponent`, `beta`
        - `is95CredibleSet`, `is99CredibleSet`
        - `posteriorProbability`, `variantId` (tag-variant id)
        - `finemappingMethod`
        - `isLead` (boolean): True when top-level `variantId` == exploded `tagVariant.variantId`
    """
    return credible_set.withColumn('tagVariant', f.explode('locus')).select(
        'studyLocusId',
        'studyId',
        'tagVariant.pValueMantissa',
        'tagVariant.pValueExponent',
        'tagVariant.beta',
        'tagVariant.is95CredibleSet',
        'tagVariant.is99CredibleSet',
        f.col('tagVariant.posteriorProbability'),
        f.col('tagVariant.variantId'),
        'finemappingMethod',
        f.when(f.col('variantId') == f.col('tagVariant.variantId'), f.lit(True))
        .otherwise(f.lit(False))
        .alias('isLead'),
    )


def process_study(study: DataFrame) -> DataFrame:
    """Normalize study metadata into the minimal fields required by downstream views.

    This function selects core study-level attributes and produces two derived columns:
      - `gwasDiseases`: a comma-joined string of `diseaseIds` when the list is non-empty,
        otherwise `NULL`.
      - `qtlStudy`: a struct containing `(geneId, biosampleId)` when `geneId` is present,
        otherwise `NULL`.

    Expected input columns (at minimum):
      - `studyId` (string): unique study identifier
      - `studyType` (string): type/category of the study
      - `diseaseIds` (array<string> or null): optional list of disease identifiers
      - `geneId` (string or null): gene id for QTL studies (optional)
      - `biosampleId` (string or null): biosample identifier used alongside `geneId`

    Notes:
      - `gwasDiseases` uses `concat_ws(',', diseaseIds)` and only produced when `size(diseaseIds) > 0`.
      - `qtlStudy` is created with `f.struct(geneId, biosampleId)` to keep related fields together;
        callers can expand it with `qtlStudy.geneId` and `qtlStudy.biosampleId`.
      - The function preserves `NULL` semantics (it returns `NULL` for the derived field when the
        source is missing), which callers may fill or filter as needed.

    Args:
        study (DataFrame): Gentropy study index.

    Returns:
        DataFrame with columns:
        - `studyId`, `studyType`
        - `gwasDiseases` (string | NULL): "|" separated disease IDs when present
        - `qtlGeneId` when `geneId` is not null
        - `qtlBiosampleId` when `biosampleId` is not null
    """
    return study.select(
        'studyId',
        'studyType',
        f.when(f.size(f.col('diseaseIds')) > 0, f.concat_ws('|', f.col('diseaseIds'))).alias('gwasDiseases'),
        f.col('geneId').alias('qtlGeneId'),
        f.col('biosampleId').alias('qtlBiosampleId'),
    )


def process_l2g(l2g: DataFrame) -> DataFrame:
    """Aggregate locus-to-gene scores and pick the top gene per locus.

    This function prepares locus->gene information for downstream VEP views by:
      - ranking `geneId` candidates per `studyLocusId` by `score` (descending),
      - keeping rows where the gene is the top-ranked hit (rank == 1) OR has a
        `score` >= 0.5 (to preserve strong secondary hits),
      - returning a compact struct column `gwasLocus2gene` containing `geneId`
        and its `locus2geneScore` for each selected locus.

    Args:
        l2g (DataFrame): gentropy locus to gene prediction dataset.

    Returns:
        DataFrame with columns:
        - `studyLocusId`
        - `gwasLocus2gene` (struct): contains
            - `geneId`
            - `locus2geneScore` (score renamed inside the struct)
    """
    return (
        l2g.select(
            'studyLocusId',
            'geneId',
            'score',
            f.rank().over(Window.partitionBy('studyLocusId').orderBy(f.col('score').desc())).alias('rank'),
        )
        .filter((f.col('rank') == 1) | (f.col('score') >= 0.5))
        .select('studyLocusId', f.col('geneId').alias('gwasGeneId'), f.col('score').alias('gwasLocusToGeneScore'))
    )


def parse_variant_id(variant_id: Column) -> tuple[Column, Column, Column, Column]:
    """Split variant identifier to chromosome, position, reference and alternate allele.

    Args:
        variant_id (Column): Valid variant identifier.

    Returns:
        tuple[Column, Column, Column, Column]:
    """
    parts = f.split(variant_id, '_')
    return (
        parts[0].alias('chromosome'),
        parts[1].cast(t.IntegerType()).alias('position'),
        parts[2].alias('referenceAllele'),
        parts[3].alias('alternateAllele'),
    )


def vep_view(
    source: dict[str, str], destination: dict[str, str], settings: dict[str, Any], properties: dict[str, str]
) -> None:
    # Initialise session:
    session = Session(app_name='VEP view generation', properties=properties)

    # Load input data:
    study = process_study(session.load_data(source['study_table']))
    credible_set = process_credible_set(session.load_data(source['credible_set']))
    l2g = process_l2g(session.load_data(path=source['l2g_table'], recursiveFileLookup=True))
    biosample = process_biosample(session.load_data(source['biosample']))

    # Join and procerss:
    vep_view = (
        credible_set.join(l2g, on='studyLocusId', how='left')
        .join(study, on='studyId', how='left')
        .join(biosample, on='qtlBiosampleId', how='left')
        .select(
            *parse_variant_id(f.col('variantId')),
            'variantId',
            'studyId',
            'studyType',
            'studyLocusId',
            'pValueMantissa',
            'pValueExponent',
            'beta',
            'is95CredibleSet',
            'is99CredibleSet',
            'posteriorProbability',
            'finemappingMethod',
            'isLead',
            'gwasGeneId',
            'gwasLocusToGeneScore',
            'gwasDiseases',
            'qtlGeneId',
            'qtlBiosampleName',
        )
        # Apply shared filter:
        .filter(f.col('position').isNotNull() & (f.length(f.col('chromosome')) < 3))
        .persist()
    )
    # Save header:
    logger.info('Writing header')
    header = '\t'.join(vep_view.columns)
    Path(destination['vep_data_header']).write_text(header + '\n')

    # Write combined data:
    logger.info('Writing data for all')
    vep_view.orderBy('chromosome', 'position').write.mode('overwrite').csv(destination['vep_data_all'], sep='\t')

    # Write GWAS data:
    logger.info('Writing data for all')
    (
        vep_view.filter(f.col('studyType') == 'gwas')
        .orderBy('chromosome', 'position')
        .write.mode('overwrite')
        .csv(destination['vep_data_gwas'], sep='\t')
    )

    # Write QTL data:
    logger.info('Writing data for all')
    (
        vep_view.filter(f.col('studyType') != 'gwas')
        .orderBy('chromosome', 'position')
        .write.mode('overwrite')
        .csv(destination['vep_data_qtl'], sep='\t')
    )
