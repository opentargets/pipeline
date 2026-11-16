"""Module to generate view for Open Target's VEP plugin."""

import io
import os
import shutil
import struct
import tempfile
import zlib
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
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
    'credibleSetId',
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

BGZF_BLOCK_SIZE = 0xFF00
BGZF_EOF = bytes.fromhex('1f8b08040000000000ff0600424302001b0003000000000000000000')
CSV_BATCH_ROWS = 500_000


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


def process_credible_set(credible_set: pl.LazyFrame, credible_set_variant: pl.LazyFrame) -> pl.LazyFrame:
    """One row per credible set variant, with its set's study, method and a lead-variant flag.

    The published variants of a credible set are its 95% members, so `is95CredibleSet` and
    `is99CredibleSet` hold for every row; `beta` is the reported effect.

    Args:
        credible_set: the published credible sets: `id`, `studyId`, `leadVariantId` and
            `fineMappingMethod`.
        credible_set_variant: their variants: `credibleSetId`, `variantId`, `pValueMantissa`,
            `pValueExponent`, `reportedBeta` and `posteriorProbability`.

    Returns:
        Lazy frame with one row per credible set variant and columns: `credibleSetId`, `studyId`,
        `pValueMantissa`, `pValueExponent`, `beta`, `is95CredibleSet`, `is99CredibleSet`,
        `posteriorProbability`, `variantId`, `finemappingMethod`, and `isLead` (True when the
        variant is the lead variant of the credible set).
    """
    lead = credible_set.select(
        pl.col('id').alias('credibleSetId'),
        'studyId',
        'leadVariantId',
        pl.col('fineMappingMethod').cast(pl.String).alias('finemappingMethod'),
    )
    return (
        credible_set_variant
        .join(lead, on='credibleSetId')
        .select(
            'credibleSetId',
            'studyId',
            'pValueMantissa',
            'pValueExponent',
            pl.col('reportedBeta').alias('beta'),
            pl.lit(True).alias('is95CredibleSet'),
            pl.lit(True).alias('is99CredibleSet'),
            'posteriorProbability',
            'variantId',
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
    """Select top locus-to-gene (L2G) predictions per credible set.

    For each `credibleSetId`, ranks candidate genes by their L2G `score` (descending)
    and retains rows where the gene is the top-ranked hit (rank == 1) OR has a score
    >= 0.5 (to preserve strong secondary hits). This allows one locus to map to multiple
    genes when evidence is strong, while avoiding noise from low-scoring candidates.

    Args:
        l2g: L2G prediction lazy frame. Expected columns: `credibleSetId`, `geneId`,
            `score`.

    Returns:
        Lazy frame with columns: `credibleSetId`, `gwasGeneId` (renamed from `geneId`),
        and `gwasLocusToGeneScore` (renamed from `score`).
    """
    return (
        l2g
        .select('credibleSetId', 'geneId', 'score')
        .with_columns(pl.col('score').rank(method='min', descending=True).over('credibleSetId').alias('rank'))
        .filter((pl.col('rank') == 1) | (pl.col('score') >= 0.5))
        .select(
            'credibleSetId',
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


def _bgzf_block(data: bytes | memoryview) -> bytes:
    """Compress one BGZF block: a gzip member whose extra field records its compressed size."""
    deflate = zlib.compressobj(zlib.Z_DEFAULT_COMPRESSION, zlib.DEFLATED, -zlib.MAX_WBITS)
    payload = deflate.compress(data) + deflate.flush()
    header = struct.pack('<4BI2BH2BHH', 31, 139, 8, 4, 0, 0, 255, 6, 66, 67, 2, len(payload) + 25)
    return header + payload + struct.pack('<2I', zlib.crc32(data), len(data))


class BgzfWriter:
    """Write a BGZF file from chunks of bytes, compressing its blocks on a shared thread pool.

    Each chunk is compressed while the next one is produced; blocks are written in order.
    """

    def __init__(self, path: Path, pool: ThreadPoolExecutor) -> None:
        self._out = path.open('wb')
        self._pool = pool
        self._pending: Iterable[bytes] = ()

    def write(self, data: bytes | memoryview) -> None:
        """Queue a chunk for compression and write the previous chunk's blocks."""
        blocks = [data[i : i + BGZF_BLOCK_SIZE] for i in range(0, len(data), BGZF_BLOCK_SIZE)]
        compressed = self._pool.map(_bgzf_block, blocks)
        self._out.writelines(self._pending)
        self._pending = compressed

    def close(self) -> None:
        """Write the remaining blocks and the BGZF end-of-file marker."""
        self._out.writelines(self._pending)
        self._out.write(BGZF_EOF)
        self._out.close()


@dataclass(frozen=True)
class TabixOutput:
    """One tabix-indexed file: where it goes, its columns and which rows it keeps."""

    destination: str
    columns: list[str]
    rows: pl.Expr = field(default_factory=lambda: pl.lit(True))


def write_tabix(frame: pl.DataFrame, order: pl.Series, outputs: list[TabixOutput], work_dir: Path) -> None:
    """Write bgzip-compressed, tabix-indexed TSV files in one pass over a frame.

    The frame is read in `order`, in batches of `CSV_BATCH_ROWS` rows, so a sorted copy of it is
    never built. Each batch is filtered and serialised for every output, and the header is written
    with the first batch. htslib can only write to the local filesystem, so the files are written
    and indexed in a temporary directory under `work_dir` and then copied to their (possibly remote)
    destinations, each index as `<destination>.tbi`.

    Args:
        frame: Data to write. Its first column is the chromosome and its second the 1-based
            position, which tabix uses as both the start and the end of the interval.
        order: Row indices that sort the frame by chromosome and position, with each chromosome
            in one contiguous block.
        outputs: The files to write.
        work_dir: Local directory for the temporary files.
    """
    work_dir.mkdir(parents=True, exist_ok=True)
    with (
        tempfile.TemporaryDirectory(dir=work_dir, prefix='vep_view_') as tmp,
        ThreadPoolExecutor(os.cpu_count() or 1) as pool,
    ):
        paths = [Path(tmp) / f'{i}.tsv.bgz' for i in range(len(outputs))]
        writers = [BgzfWriter(path, pool) for path in paths]
        for offset in range(0, max(order.len(), 1), CSV_BATCH_ROWS):
            batch = frame[order.slice(offset, CSV_BATCH_ROWS)]
            for output, writer in zip(outputs, writers, strict=True):
                buffer = io.BytesIO()
                batch.filter(output.rows).select(output.columns).write_csv(
                    buffer, separator='\t', include_header=offset == 0
                )
                writer.write(buffer.getbuffer())
        for writer in writers:
            writer.close()

        def index_and_upload(path: Path, output: TabixOutput) -> None:
            pysam.tabix_index(str(path), force=True, seq_col=0, start_col=1, end_col=1, meta_char='#', zerobased=False)
            _upload(path, output.destination)
            _upload(Path(f'{path}.tbi'), f'{output.destination}.tbi')

        list(pool.map(index_and_upload, paths, outputs))


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
    - `vep_data_qtl`: QTL-only rows (studyType ending in 'qtl'), without the GWAS columns.

    The header line starts with `#`, so tabix keeps it as a meta line and `tabix -H` returns it.

    Variant IDs (format `chr_pos_ref_alt`) are parsed into separate `chromosome`,
    `position`, `referenceAllele`, and `alternateAllele` columns. Rows with a null
    position or a chromosome name longer than 2 characters (e.g. mitochondrial or patch
    contigs) are excluded.

    Args:
        source: Mapping of input dataset keys to their parquet directory paths.
            Required keys: `credible_set`, `credible_set_variant`, `study_table`, `l2g_table`, `biosample`.
        destination: Mapping of output keys to file paths.
            Required keys: `vep_data_all`, `vep_data_gwas`, `vep_data_qtl`. Each tabix index is
            written next to its data file, with `.tbi` appended to the path.
        settings: Unused transformer settings dict (reserved for future use).
        config: Otter pipeline config object; its `work_path` holds the temporary files.
    """
    logger.info('Loading input data')
    credible_set = process_credible_set(
        scan_dataset(source['credible_set']), scan_dataset(source['credible_set_variant'])
    )
    study = process_study(scan_dataset(source['study_table']))
    l2g = process_l2g(scan_dataset(source['l2g_table']))
    biosample = process_biosample(scan_dataset(source['biosample']))

    logger.info('Joining and processing data')
    result = (
        credible_set
        .join(l2g, on='credibleSetId', how='left')
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
        .collect()
    )
    order = result.select('#chromosome', 'position').with_row_index().sort('#chromosome', 'position')['index']

    logger.info('Writing datasets')
    write_tabix(
        result,
        order,
        [
            TabixOutput(destination['vep_data_all'], [*COMMON_COLUMNS, *GWAS_COLUMNS, *QTL_COLUMNS]),
            TabixOutput(destination['vep_data_gwas'], [*COMMON_COLUMNS, *GWAS_COLUMNS], pl.col('studyType') == 'gwas'),
            TabixOutput(
                destination['vep_data_qtl'],
                [*COMMON_COLUMNS, *QTL_COLUMNS],
                pl.col('studyType').str.ends_with('qtl'),
            ),
        ],
        Path(config.work_path),
    )
