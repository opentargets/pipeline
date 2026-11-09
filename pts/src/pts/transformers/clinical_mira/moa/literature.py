"""Selection constants and the sentence pass for the literature mechanism extraction."""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import fsspec
import polars as pl
from loguru import logger

from pts.transformers.clinical_mira import CATEGORY_RANKS_STR, ClinicalStageCategory

PER_DRUG = 25
PER_FILE_CAP = 300
MIN_CHARS = 30
MAX_CHARS = 1200
SECTION_RANK = {'title': 0, 'abstract': 1, 'intro': 2, 'concl': 3, 'discuss': 4, 'results': 5}
CUE = (
    r'(?i)\b(inhibitor|inhibits|inhibition|agonist|antagonist|blocker|blocks|modulator|activator|targeting|targets'
    r'|binds|binding|degrader|antibody|siRNA|antisense|selective)\b'
)

MATCH_COLUMNS = ['pmid', 'pmcid', 'section', 'year', 'type', 'mappedId', 'label', 'text', 'isValid']
CD_COLUMNS = ['parent', 'mappedId', 'pub', 'pmid', 'pmcid', 'section', 'year', 'label', 'text', 'h', 'cue', 'srank']
HASH_SEED = 7


def clinical_parents(drug_molecule: pl.DataFrame) -> pl.DataFrame:
    """Map molecule ids to their parent, keeping the parents that reached the clinic.

    A parent is a molecule without ``parentId``. It qualifies when its ``maximumClinicalStage``
    lies between ``EARLY_PHASE_1`` and ``APPROVAL`` inclusive.

    Args:
        drug_molecule: Frame with ``id``, ``parentId`` and ``maximumClinicalStage``.

    Returns:
        Frame with ``id`` and ``parent``, one row per molecule (parents map to themselves) whose
        parent qualifies.
    """
    newest = CATEGORY_RANKS_STR[ClinicalStageCategory.APPROVAL.value]
    oldest = CATEGORY_RANKS_STR[ClinicalStageCategory.EARLY_PHASE_1.value]
    qualifying = (
        drug_molecule
        .filter(pl.col('parentId').is_null())
        .filter(
            pl
            .col('maximumClinicalStage')
            .replace_strict(CATEGORY_RANKS_STR, default=None, return_dtype=pl.Int64)
            .is_between(newest, oldest)
        )
        .select(parent='id')
    )
    return (
        drug_molecule
        .select('id', parent=pl.coalesce('parentId', 'id'))
        .join(qualifying, on='parent')
        .select('id', 'parent')
    )


def pass_file(match: pl.DataFrame, parents: pl.DataFrame) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Reduce one literature match file to drug sentences and gene-sentence pairs.

    Args:
        match: Rows of one match file with the ``MATCH_COLUMNS`` columns.
        parents: Frame with ``id`` and ``parent``, from ``clinical_parents``.

    Returns:
        ``(cd, gp)``. ``cd`` holds the valid drug mentions of clinical parents, at most
        ``PER_FILE_CAP`` per parent, cue sentences first, then by section rank and hash, with the
        ``CD_COLUMNS`` columns. ``gp`` holds the distinct ``pub``, ``h``, ``gene`` of valid gene mentions.
    """
    t = match.filter(pl.col('isValid') & pl.col('type').is_in(['CD', 'GP'])).with_columns(
        pub=pl.coalesce('pmid', 'pmcid'), h=pl.col('text').hash(seed=HASH_SEED)
    )
    gp = t.filter(pl.col('type') == 'GP').select('pub', 'h', gene='mappedId').unique()
    cd = (
        t
        .filter(pl.col('type') == 'CD')
        .join(parents, left_on='mappedId', right_on='id')
        .with_columns(
            cue=pl.col('text').str.contains(CUE),
            srank=pl
            .col('section')
            .str.split(',')
            .list.first()
            .replace_strict(SECTION_RANK, default=9, return_dtype=pl.Int8),
        )
        .sort('parent', pl.col('cue'), 'srank', 'h', descending=[False, True, False, False])
        .filter(pl.int_range(pl.len()).over('parent') < PER_FILE_CAP)
        .select(CD_COLUMNS)
    )
    return cd, gp


def _atomic_write(df: pl.DataFrame, path: Path) -> None:
    tmp = path.with_suffix('.tmp')
    df.write_parquet(tmp)
    tmp.rename(path)


def sentence_pass(
    match_uri: str,
    parents: pl.DataFrame,
    workers: int,
    scratch: Path,
    storage_options: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    """Run ``pass_file`` over every parquet file of a literature match directory.

    Each file is read with the ``MATCH_COLUMNS`` columns only and written to
    ``<scratch>/cd/<name>.parquet`` and ``<scratch>/gp/<name>.parquet``, ``name`` being the
    file's name without extension. A file whose two outputs exist is skipped.

    Args:
        match_uri: Directory (local or ``gs://``) holding the match ``*.parquet`` files.
        parents: Frame with ``id`` and ``parent``, from ``clinical_parents``.
        workers: Number of files processed concurrently.
        scratch: Local directory receiving the outputs.
        storage_options: Options of the fsspec filesystem serving ``match_uri``, for example
            ``{'requester_pays': True, 'project': ...}`` on GCS.

    Returns:
        The ``cd`` and ``gp`` directories.

    Raises:
        ValueError: If ``match_uri`` holds no parquet files.
    """
    fs, root = fsspec.core.url_to_fs(match_uri, **(storage_options or {}))
    files = sorted(fs.glob(root.rstrip('/') + '/*.parquet'))
    if not files:
        raise ValueError(f'no parquet files found under {match_uri}')
    cd_dir, gp_dir = scratch / 'cd', scratch / 'gp'
    cd_dir.mkdir(parents=True, exist_ok=True)
    gp_dir.mkdir(parents=True, exist_ok=True)

    def one(path: str) -> None:
        name = Path(path).stem
        cd_out, gp_out = cd_dir / f'{name}.parquet', gp_dir / f'{name}.parquet'
        if cd_out.exists() and gp_out.exists():
            return
        started = time.time()
        with fs.open(path, 'rb') as handle:
            match = pl.read_parquet(handle, columns=MATCH_COLUMNS)
        cd, gp = pass_file(match, parents)
        _atomic_write(gp, gp_out)
        _atomic_write(cd, cd_out)
        logger.info(f'{name}: {match.height} rows, {cd.height} cd, {gp.height} gp in {time.time() - started:.0f}s')

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(one, files))
    return cd_dir, gp_dir


def join_genes(cd_dir: Path, gp_dir: Path) -> pl.DataFrame:
    """Attach the genes mentioned in each drug sentence.

    Args:
        cd_dir: Directory of per-file ``cd`` parquet files.
        gp_dir: Directory of per-file ``gp`` parquet files.

    Returns:
        The ``cd`` rows with a ``genes`` column: the sorted unique genes of the same
        ``pub`` and ``h``, or null when there are none.
    """
    cd = pl.scan_parquet(cd_dir / '*.parquet')
    keys = cd.select('pub', 'h').unique()
    gp = (
        pl
        .scan_parquet(gp_dir / '*.parquet')
        .join(keys, on=['pub', 'h'], how='semi')
        .group_by('pub', 'h')
        .agg(genes=pl.col('gene').unique().sort())
    )
    return cd.join(gp, on=['pub', 'h'], how='left').collect(engine='streaming')
