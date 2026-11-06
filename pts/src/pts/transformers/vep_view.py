"""Module to generate view for Open Target's VEP plugin."""

import shutil
import tempfile
from pathlib import Path
from typing import Any

import polars as pl
import pysam
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import scan_dataset

# Columns shared by every VEP file. The leading `#` on the first column makes the header line a
# tabix meta line, so it is kept in the index and `tabix -H` returns the column names.
COMMON_COLUMNS = [
    '#chromosome',
    'position',
    'referenceAllele',
    'alternateAllele',
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
]
GWAS_COLUMNS = ['gwasGeneId', 'gwasLocusToGeneScore', 'gwasDiseases']
QTL_COLUMNS = ['qtlGeneId', 'qtlBiosampleName']


def process_biosample(biosample: pl.LazyFrame) -> pl.LazyFrame:
    """Select and rename biosample identifier and display name for QTL joins.

    Renames `biosampleId` → `qtlBiosampleId` and `biosampleName` → `qtlBiosampleName`
    to match the naming convention used after joining with the credible-set/study data,
    and deduplicates rows.

    Args:
        biosample: Biosample index lazy frame.

    Returns:
        Deduplicated lazy frame with columns `qtlBiosampleId` and `qtlBiosampleName`.
    """
    return biosample.select(
        pl.col('biosampleId').alias('qtlBiosampleId'),
        pl.col('biosampleName').alias('qtlBiosampleName'),
    ).unique()


def process_credible_set(credible_set: pl.LazyFrame) -> pl.LazyFrame:
    """Flatten a credible set into one row per tag variant and compute a lead-variant flag.

    The credible set stores per-locus finemapping results with a nested `locus` array of
    structs, one entry per tag variant. This function explodes that array and unnests the
    struct so each tag variant occupies its own row. The lead variant is identified by
    comparing the top-level `variantId` (the lead) to the exploded per-row `variantId`.

    Args:
        credible_set: Credible set lazy frame. Expected columns include `studyLocusId`,
            `studyId`, `variantId` (lead), `finemappingMethod`, and `locus` (array of
            structs with fields `variantId`, `pValueMantissa`, `pValueExponent`, `beta`,
            `is95CredibleSet`, `is99CredibleSet`, `posteriorProbability`).

    Returns:
        Lazy frame with one row per tag variant and columns: `studyLocusId`, `studyId`,
        `pValueMantissa`, `pValueExponent`, `beta`, `is95CredibleSet`, `is99CredibleSet`,
        `posteriorProbability`, `variantId` (tag variant), `finemappingMethod`, and
        `isLead` (True when the tag variant is the lead variant of the locus).
    """
    return (
        credible_set
        .select('studyLocusId', 'studyId', pl.col('variantId').alias('leadVariantId'), 'locus', 'finemappingMethod')
        .explode('locus', empty_as_null=True)
        .unnest('locus')
        .select(
            'studyLocusId',
            'studyId',
            'pValueMantissa',
            'pValueExponent',
            'beta',
            'is95CredibleSet',
            'is99CredibleSet',
            'posteriorProbability',
            pl.col('variantId'),
            'finemappingMethod',
            (pl.col('leadVariantId') == pl.col('variantId')).alias('isLead'),
        )
        .unique()
    )


def process_study(study: pl.LazyFrame) -> pl.LazyFrame:
    """Normalize study metadata into the minimal fields required by downstream views.

    Selects core study-level attributes and produces two derived columns:
    - `gwasDiseases`: pipe-joined string of `diseaseIds` when the list is non-empty,
      otherwise NULL. Used to annotate GWAS loci with associated disease identifiers.
    - `qtlGeneId` / `qtlBiosampleId`: renamed from `geneId` / `biosampleId` to reflect
      that these fields are only populated for QTL studies.

    Args:
        study: Study index lazy frame. Expected columns include `studyId`, `studyType`,
            `diseaseIds` (list of strings), `geneId`, and `biosampleId`.

    Returns:
        Lazy frame with columns: `studyId`, `studyType`, `gwasDiseases` (string | null),
        `qtlGeneId`, and `qtlBiosampleId`.
    """
    return study.select(
        'studyId',
        'studyType',
        pl.when(pl.col('diseaseIds').list.len() > 0).then(pl.col('diseaseIds').list.join('|')).alias('gwasDiseases'),
        pl.col('geneId').alias('qtlGeneId'),
        pl.col('biosampleId').alias('qtlBiosampleId'),
    )


def process_l2g(l2g: pl.LazyFrame) -> pl.LazyFrame:
    """Select top locus-to-gene (L2G) predictions per study locus.

    For each `studyLocusId`, ranks candidate genes by their L2G `score` (descending)
    and retains rows where the gene is the top-ranked hit (rank == 1) OR has a score
    >= 0.5 (to preserve strong secondary hits). This allows one locus to map to multiple
    genes when evidence is strong, while avoiding noise from low-scoring candidates.

    Args:
        l2g: L2G prediction lazy frame. Expected columns: `studyLocusId`, `geneId`,
            `score`.

    Returns:
        Lazy frame with columns: `studyLocusId`, `gwasGeneId` (renamed from `geneId`),
        and `gwasLocusToGeneScore` (renamed from `score`).
    """
    return (
        l2g
        .select('studyLocusId', 'geneId', 'score')
        .with_columns(pl.col('score').rank(method='min', descending=True).over('studyLocusId').alias('rank'))
        .filter((pl.col('rank') == 1) | (pl.col('score') >= 0.5))
        .select(
            'studyLocusId',
            pl.col('geneId').alias('gwasGeneId'),
            pl.col('score').alias('gwasLocusToGeneScore'),
        )
    )


def _upload(local_path: Path, destination: str) -> None:
    """Copy a local file to a destination that may be local or remote.

    A plain `open()` on a `gs://` path would write to the container's disk instead, so the copy
    goes through `StorageHandle`.
    """
    with local_path.open('rb') as src, StorageHandle(destination).open('wb') as dst:
        shutil.copyfileobj(src, dst)


def write_tabix(frame: pl.DataFrame, destination: str, work_dir: Path) -> None:
    """Write a frame as a bgzip-compressed TSV with a tabix index.

    htslib can only write to the local filesystem, so the TSV is written, compressed and indexed
    in a temporary directory under `work_dir` and then copied to the (possibly remote)
    destinations. The index is written next to the data file as `<destination>.tbi`, the name
    tabix readers look for. The frame must be sorted by `#chromosome` and `position`, with each
    chromosome in one contiguous block.

    Args:
        frame: Data to write. The first column is the chromosome and the second the 1-based
            position, which tabix uses as both the start and the end of the interval.
        destination: Where to write the `.tsv.bgz` file.
        work_dir: Local directory for the temporary files.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_dir, prefix='vep_view_') as tmp:
        tsv = Path(tmp) / 'data.tsv'
        bgz = Path(tmp) / 'data.tsv.bgz'
        frame.write_csv(tsv, separator='\t')
        pysam.tabix_compress(str(tsv), str(bgz), force=True)
        tsv.unlink()
        pysam.tabix_index(str(bgz), force=True, seq_col=0, start_col=1, end_col=1, meta_char='#', zerobased=False)
        _upload(bgz, destination)
        _upload(Path(f'{bgz}.tbi'), f'{destination}.tbi')


def vep_view(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Build and write the VEP plugin input view from Open Targets genetics datasets.

    This dataset describes every single variant represented in an Open Targets credible set.
    For each variant, VEP describes which credible sets it belongs to, its fine-mapping statistics,
    and the gene(s) it likely affects. For GWAS loci, the gene comes from a Locus-to-Gene prediction;
    for QTL loci, it is the directly measured effector gene together with the tissue/biosample context.

    Loads four input datasets (credible sets, study index, L2G predictions, biosample
    index), processes and joins them into a flat, per-tag-variant table, and writes three
    bgzip-compressed, tabix-indexed TSV files, sorted by chromosome/position:
    - `vep_data_all`: all study types combined, with both GWAS and QTL columns.
    - `vep_data_gwas`: GWAS-only rows (studyType == 'gwas'), without the QTL columns.
    - `vep_data_qtl`: QTL-only rows (studyType != 'gwas'), without the GWAS columns.

    The header line starts with `#`, so tabix keeps it as a meta line and `tabix -H` returns it.

    Variant IDs (format `chr_pos_ref_alt`) are parsed into separate `chromosome`,
    `position`, `referenceAllele`, and `alternateAllele` columns. Rows with a null
    position or a chromosome name longer than 2 characters (e.g. mitochondrial or patch
    contigs) are excluded.

    Args:
        source: Mapping of input dataset keys to their parquet directory paths.
            Required keys: `credible_set`, `study_table`, `l2g_table`, `biosample`.
        destination: Mapping of output keys to file paths.
            Required keys: `vep_data_all`, `vep_data_gwas`, `vep_data_qtl`. Each tabix index is
            written next to its data file, with `.tbi` appended to the path.
        settings: Unused transformer settings dict (reserved for future use).
        config: Otter pipeline config object; its `work_path` holds the temporary files.
    """
    logger.info('Loading input data')
    credible_set = process_credible_set(scan_dataset(source['credible_set']))
    study = process_study(scan_dataset(source['study_table']))
    l2g = process_l2g(scan_dataset(source['l2g_table']))
    biosample = process_biosample(scan_dataset(source['biosample']))

    logger.info('Joining and processing data')
    result = (
        credible_set
        .join(l2g, on='studyLocusId', how='left')
        .join(study, on='studyId', how='left')
        .join(biosample, on='qtlBiosampleId', how='left')
        .with_columns(
            pl.col('variantId').str.split('_').list.get(0).alias('#chromosome'),
            pl.col('variantId').str.split('_').list.get(1).cast(pl.Int32, strict=False).alias('position'),
            pl.col('variantId').str.split('_').list.get(2).alias('referenceAllele'),
            pl.col('variantId').str.split('_').list.get(3).alias('alternateAllele'),
        )
        .select(*COMMON_COLUMNS, *GWAS_COLUMNS, *QTL_COLUMNS)
        .filter(
            pl.col('position').is_not_null(),
            pl.col('#chromosome').str.len_chars() < 3,
        )
        .sort('#chromosome', 'position')
        .collect()
    )

    work_dir = Path(config.work_path)

    logger.info('Writing full dataset')
    write_tabix(result, destination['vep_data_all'], work_dir)

    logger.info('Writing GWAS dataset')
    write_tabix(
        result.filter(pl.col('studyType') == 'gwas').select(*COMMON_COLUMNS, *GWAS_COLUMNS),
        destination['vep_data_gwas'],
        work_dir,
    )

    logger.info('Writing QTL dataset')
    write_tabix(
        result.filter(pl.col('studyType') != 'gwas').select(*COMMON_COLUMNS, *QTL_COLUMNS),
        destination['vep_data_qtl'],
        work_dir,
    )
