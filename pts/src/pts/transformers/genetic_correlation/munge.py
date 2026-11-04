"""Each representative study's summary statistics joined to its ancestry's LD scores, once.

A pair's regression needs the two studies' effects on the SNPs the LD panel covers. Joining every
study to the panel here, and keeping only the columns the regression reads, makes each of the
hundreds of thousands of pairs a join of two small files.

Summary statistics live in a bucket that may be requester pays, so the parts are fetched through
otter with a billing project rather than scanned in place.
"""

from __future__ import annotations

import shutil
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.storage_context import storage_context
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.genetic_correlation.settings import MungeDestination, MungeSettings, MungeSource
from pts.transformers.utils.dataset import check_destinations_absent, list_parts, write_dataset

COPY_BYTES = 16 * 1024 * 1024

VARIANT_KEY = ['chromosome', 'position', 'ref', 'alt']

MUNGED_SCHEMA = pl.Schema({
    'chromosome': pl.String,
    'position': pl.Int64,
    'ref': pl.String,
    'alt': pl.String,
    'beta': pl.Float64,
    'standardError': pl.Float64,
    'sampleSize': pl.Float64,
    'ldScore': pl.Float64,
})
"""One munged study: its effects on the panel's SNPs, in genomic order."""

PANELS_SCHEMA = pl.Schema({'ldAncestry': pl.String, 'mSnps': pl.Int64})

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


def read_panel(path: str) -> pl.DataFrame:
    """One ancestry's LD scores, a tab separated file or directory of parts, keyed by `VARIANT_KEY`.

    Returns:
        `chromosome`, `position`, `ref`, `alt`, `ldScore` and, when the panel has one, `alleleFrequency`.
    """
    parts = list_parts(path, '*.csv*')
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
    return scan.select(selected).filter(pl.col('ldScore').is_not_null()).unique(VARIANT_KEY).collect()


def panel_size(panel: pl.DataFrame, maf_threshold: float) -> int:
    """The number of panel SNPs that count towards M: those common enough, or all without frequencies."""
    if 'alleleFrequency' not in panel.columns:
        return panel.height
    common = pl.col('alleleFrequency').is_between(maf_threshold, 1 - maf_threshold)
    return panel.filter(common).height


def munge_study(sumstats: pl.LazyFrame, fallback_sample_size: float | None, panel: pl.DataFrame) -> pl.DataFrame:
    """Join one study's summary statistics to the panel, in `MUNGED_SCHEMA`.

    Alleles come from `variantId` (`chr_pos_ref_alt`). A variant without a sample size of its own
    takes the study's effective one. Variants with no effect, no error, or a duplicate key are dropped.
    """
    alleles = pl.col('variantId').str.split('_')
    key = pl.col('chromosome').cast(pl.String)
    return (
        sumstats
        .select(
            key,
            pl.col('position').cast(pl.Int64),
            alleles.list.get(2, null_on_oob=True).alias('ref'),
            alleles.list.get(3, null_on_oob=True).alias('alt'),
            pl.col('beta').cast(pl.Float64),
            pl.col('standardError').cast(pl.Float64),
            pl.coalesce(pl.col('sampleSize').cast(pl.Float64), pl.lit(fallback_sample_size, dtype=pl.Float64)).alias(
                'sampleSize'
            ),
        )
        .filter(
            pl.col('beta').is_not_null(),
            pl.col('standardError') > 0,
            pl.col('sampleSize').is_not_null(),
            pl.col('ref').is_not_null(),
            pl.col('alt').is_not_null(),
        )
        .unique(VARIANT_KEY)
        .join(panel.lazy().select(*VARIANT_KEY, 'ldScore'), on=VARIANT_KEY)
        .sort('chromosome', 'position')
        .select(list(MUNGED_SCHEMA))
        .collect()
    )


def _munge_one(
    study: dict[str, Any], panel: pl.DataFrame, sources: MungeSource, destinations: MungeDestination, scratch: Path
) -> int:
    """Fetch, munge and publish one study; returns its SNP count."""
    study_id, ancestry = study['studyId'], study['ldAncestry']
    local = scratch / ancestry / study_id
    parts = [
        fetch(part, local / 'parts' / Path(part).name)
        for part in list_parts(f'{sources.sumstats}/{study_id}', '*.parquet')
    ]
    munged = munge_study(pl.scan_parquet(parts), study['effectiveSampleSize'], panel)
    local.mkdir(parents=True, exist_ok=True)
    out = local / f'{study_id}.parquet'
    munged.write_parquet(out, compression='zstd')
    publish(out, f'{destinations.munged}/{ancestry}/{study_id}.parquet')
    shutil.rmtree(local, ignore_errors=True)
    return munged.height


def munge(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Munge every representative study.

    Args:
        source: the `MungeSource` paths.
        destination: the `MungeDestination` paths.
        settings: the `MungeSettings`.
        config: for `work_path`, where the fetched parts go.
    """
    sources = MungeSource.model_validate(source)
    destinations = MungeDestination.model_validate(destination)
    options = MungeSettings.model_validate(settings)
    check_destinations_absent([destinations.munged, destinations.panels])
    studies = (
        pl
        .scan_parquet(list_parts(sources.representatives, '*.parquet'))
        .select('studyId', 'ldAncestry', 'effectiveSampleSize')
        .unique()
        .sort('ldAncestry', 'studyId')
        .collect()
    )
    panels: list[dict[str, Any]] = []
    with (
        TemporaryDirectory(dir=config.work_path) as directory,
        storage_context(billing_project=options.billing_project),
    ):
        scratch = Path(directory)
        for (ancestry,), group in studies.group_by('ldAncestry', maintain_order=True):
            panel = read_panel(f'{sources.ldscore}/{options.ldscore_template.format(ancestry=ancestry)}')
            m_snps = panel_size(panel, options.maf_threshold)
            panels.append({'ldAncestry': ancestry, 'mSnps': m_snps})
            logger.info('{}: {} panel SNPs, M = {}; munging {} studies', ancestry, panel.height, m_snps, group.height)
            work = partial(_munge_one, panel=panel, sources=sources, destinations=destinations, scratch=scratch)
            with ThreadPoolExecutor(max_workers=options.parallel_studies) as pool:
                counts = list(pool.map(work, group.iter_rows(named=True)))
            logger.info('{}: {} SNPs over {} studies', ancestry, sum(counts), len(counts))
    write_dataset(pl.DataFrame(panels, schema=PANELS_SCHEMA), destinations.panels)
