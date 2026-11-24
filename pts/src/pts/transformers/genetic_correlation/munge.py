"""Each representative study's summary statistics joined to its ancestry's LD scores, once.

`munge_one` is one unit of work: fetch a study's summary statistics, join them to the panel and
write one parquet file. `pts.pyspark.genetic_correlation_munge` runs it for every representative.

Summary statistics live in a bucket that may be requester pays, so the parts are fetched through
otter, which bills the project of the surrounding `storage_context`, rather than scanned in place.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.utils.dataset import list_parts

COPY_BYTES = 16 * 1024 * 1024

MUNGE_VERSION = 2
"""The version of the munged format and of how it is made: a run reuses a previous run's munged
studies only if both were made with the same version. Increment it with any change to
`munge_study` or `read_panel` that changes what a munged study holds."""

VARIANT_KEY = ['chromosome', 'position', 'ref', 'alt']

MUNGED_SCHEMA = pl.Schema({
    'snp': pl.UInt32,
    'z': pl.Float64,
    'sampleSize': pl.Float64,
    'ldScore': pl.Float64,
})
"""One munged study: its z-scores on the panel's SNPs, aligned to the panel's alleles. `snp` is the
SNP's row in the panel (`read_panel`), so sorting by it is genomic order and two studies of one
ancestry join on it."""

STUDIES_SCHEMA = pl.Schema({
    'studyId': pl.String,
    'ldAncestry': pl.String,
    'nSnps': pl.Int64,
    'isUsable': pl.Boolean,
    'isReused': pl.Boolean,
})
"""Each munged study's number of panel SNPs, whether it has enough to be correlated, and whether it
was copied from the previous run rather than munged."""

PANELS_SCHEMA = pl.Schema({
    'ldAncestry': pl.String,
    'mSnps': pl.Int64,
    'panelChecksum': pl.String,
    'mungeVersion': pl.Int64,
})
"""Each panel's M, the checksum of its files, and the `MUNGE_VERSION` its studies were munged with."""

_PANEL_COLUMNS = {
    'CHR': pl.String,
    'BP_hg38': pl.Int64,
    'ref': pl.String,
    'alt': pl.String,
    'L2': pl.Float64,
    'AF': pl.Float64,
}


def is_remote(path: str) -> bool:
    return '://' in path


def fetch(path: str, local: Path) -> Path:
    """Copy one file to `local` and return it; a local `path` is returned as is."""
    if not is_remote(path):
        return Path(path)
    local.parent.mkdir(parents=True, exist_ok=True)
    with StorageHandle(path).open('rb') as source, local.open('wb') as target:
        shutil.copyfileobj(source, target, COPY_BYTES)
    return local


def publish(local: Path, target: str) -> None:
    """Move a local file to `target`, a local path or an object store URI."""
    if not is_remote(target):
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        shutil.move(local, target)
        return
    with local.open('rb') as source, StorageHandle(target).open('wb') as out:
        shutil.copyfileobj(source, out, COPY_BYTES)
    local.unlink()


def read_panel(path: str, scratch: Path) -> pl.DataFrame:
    """One ancestry's LD scores, a tab separated file or directory of parts, keyed by `VARIANT_KEY`.

    Args:
        path: the panel, local or remote.
        scratch: where the parts of a remote panel are copied to be read.

    Returns:
        `snp`, the row in genomic order, then `chromosome`, `position`, `ref`, `alt`, `ldScore` and,
        when the panel has one, `alleleFrequency`.
    """
    parts = [fetch(part, scratch / Path(part).name) for part in list_parts(path, '*.csv*')]
    scan = pl.scan_csv(parts, separator='\t', schema_overrides=_PANEL_COLUMNS)
    columns = scan.collect_schema().names()
    position = 'BP_hg38' if 'BP_hg38' in columns else 'BP'
    selected = [
        pl.col('CHR').alias('chromosome'),
        pl.col(position).cast(pl.Int64).alias('position'),
        'ref',
        'alt',
        pl.col('L2').alias('ldScore'),
    ]
    if 'AF' in columns:
        selected.append(pl.col('AF').alias('alleleFrequency'))
    return (
        scan
        .select(selected)
        .filter(pl.col('ldScore').is_not_null())
        .unique(VARIANT_KEY)
        .sort(VARIANT_KEY)
        .with_row_index('snp')
        .collect()
    )


def panel_checksum(path: str) -> str:
    """The SHA-256 of a panel's files, in the order `read_panel` reads them."""
    digest = hashlib.sha256()
    for part in list_parts(path, '*.csv*'):
        with StorageHandle(part).open('rb') if is_remote(part) else Path(part).open('rb') as handle:
            while block := handle.read(COPY_BYTES):
                digest.update(block)
    return digest.hexdigest()


def copy(source: str, target: str, scratch: Path) -> None:
    """Copy one file between local paths or object store URIs, through `scratch`."""
    local = fetch(source, scratch / Path(source).name)
    if not is_remote(source):
        staged = scratch / f'{Path(source).name}.copy'
        shutil.copyfile(local, staged)
        local = staged
    publish(local, target)


def panel_size(panel: pl.DataFrame, maf_threshold: float) -> int:
    """The number of panel SNPs that count towards M: those common enough, or all without frequencies."""
    if 'alleleFrequency' not in panel.columns:
        return panel.height
    common = pl.col('alleleFrequency').is_between(maf_threshold, 1 - maf_threshold)
    return panel.filter(common).height


def munge_study(sumstats: pl.LazyFrame, fallback_sample_size: float | None, panel: pl.DataFrame) -> pl.DataFrame:
    """Align one study's summary statistics to the panel, in `MUNGED_SCHEMA`.

    Alleles come from `variantId` (`chr_pos_ref_alt`). A variant whose alleles are the panel's
    swapped is kept with its z-score negated; one that matches both ways keeps the direct match.
    A variant without a sample size of its own takes the study's effective one. Variants with no
    effect, no error, or a duplicate key are dropped.
    """
    alleles = pl.col('variantId').str.split('_')
    study = (
        sumstats
        .select(
            pl.col('chromosome').cast(pl.String),
            pl.col('position').cast(pl.Int64),
            alleles.list.get(2, null_on_oob=True).alias('ref'),
            alleles.list.get(3, null_on_oob=True).alias('alt'),
            (pl.col('beta').cast(pl.Float64) / pl.col('standardError').cast(pl.Float64)).alias('z'),
            pl.coalesce(pl.col('sampleSize').cast(pl.Float64), pl.lit(fallback_sample_size, dtype=pl.Float64)).alias(
                'sampleSize'
            ),
            pl.col('standardError').cast(pl.Float64).alias('_se'),
        )
        .filter(
            pl.col('z').is_finite(),
            pl.col('_se') > 0,
            pl.col('sampleSize').is_not_null(),
            pl.col('ref').is_not_null(),
            pl.col('alt').is_not_null(),
        )
        .drop('_se')
        .unique(VARIANT_KEY)
    )
    keys = panel.lazy().select('snp', *VARIANT_KEY, 'ldScore')
    swapped = keys.rename({'ref': 'alt', 'alt': 'ref'})
    return (
        pl
        .concat([
            study.join(keys, on=VARIANT_KEY).with_columns(pl.lit(0).alias('_swapped')),
            study.join(swapped, on=VARIANT_KEY).with_columns(-pl.col('z'), pl.lit(1).alias('_swapped')),
        ])
        .sort('snp', '_swapped')
        .unique('snp', keep='first', maintain_order=True)
        .select(list(MUNGED_SCHEMA))
        .cast(MUNGED_SCHEMA)
        .collect()
    )


def munge_one(study: dict[str, Any], panel: pl.DataFrame, sumstats: str, munged: str, scratch: Path) -> int:
    """Fetch, munge and write one study.

    Args:
        study: `studyId`, `ldAncestry` and `effectiveSampleSize`, a row of the representatives.
        panel: the study's ancestry's panel, from `read_panel`.
        sumstats: the directory of every study's summary statistics, one dataset per `studyId`.
        munged: the directory the study is written to, as `<ldAncestry>/<studyId>.parquet`.
        scratch: a local directory for the fetched parts.

    Returns:
        The study's number of SNPs on the panel.
    """
    study_id, ancestry = study['studyId'], study['ldAncestry']
    local = scratch / ancestry / study_id
    parts = [
        fetch(part, local / 'parts' / Path(part).name) for part in list_parts(f'{sumstats}/{study_id}', '*.parquet')
    ]
    frame = munge_study(pl.scan_parquet(parts), study['effectiveSampleSize'], panel)
    local.mkdir(parents=True, exist_ok=True)
    out = local / f'{study_id}.parquet'
    frame.write_parquet(out, compression='zstd')
    publish(out, f'{munged}/{ancestry}/{study_id}.parquet')
    shutil.rmtree(local, ignore_errors=True)
    return frame.height
