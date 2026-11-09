"""Selection constants, the sentence pass and the deterministic selection for the literature mechanism extraction."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

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


def _parquet_files(uri: str) -> list[str]:
    """The sorted ``*.parquet`` files directly under ``uri``.

    Raises:
        ValueError: If ``uri`` holds no parquet files.
    """
    files = sorted(StorageHandle(uri).glob('*.parquet'))
    if not files:
        raise ValueError(f'no parquet files found under {uri}')
    return files


def read_parquet_dir(uri: str, columns: list[str]) -> pl.DataFrame:
    """Read ``columns`` of every parquet file of a directory into one frame, in file name order.

    Args:
        uri: Directory (local or ``gs://``) holding ``*.parquet`` files.
        columns: The columns to read.

    Returns:
        The rows of every file.

    Raises:
        ValueError: If ``uri`` holds no parquet files.
    """
    return pl.concat([pl.read_parquet(path, columns=columns) for path in _parquet_files(uri)])


def sentence_pass(match_uri: str, parents: pl.DataFrame, workers: int, scratch: Path) -> tuple[Path, Path]:
    """Run ``pass_file`` over every parquet file of a literature match directory.

    Each file is read with the ``MATCH_COLUMNS`` columns only and written to
    ``<scratch>/cd/<name>.parquet`` and ``<scratch>/gp/<name>.parquet``, ``name`` being the
    file's name without extension. A file whose two outputs exist is skipped.

    Args:
        match_uri: Directory (local or ``gs://``) holding the match ``*.parquet`` files.
        parents: Frame with ``id`` and ``parent``, from ``clinical_parents``.
        workers: Number of files processed concurrently.
        scratch: Local directory receiving the outputs.

    Returns:
        The ``cd`` and ``gp`` directories.

    Raises:
        ValueError: If ``match_uri`` holds no parquet files.
    """
    files = _parquet_files(match_uri)
    cd_dir, gp_dir = scratch / 'cd', scratch / 'gp'
    cd_dir.mkdir(parents=True, exist_ok=True)
    gp_dir.mkdir(parents=True, exist_ok=True)

    def one(path: str) -> None:
        name = Path(path).stem
        cd_out, gp_out = cd_dir / f'{name}.parquet', gp_dir / f'{name}.parquet'
        if cd_out.exists() and gp_out.exists():
            return
        started = time.time()
        match = pl.read_parquet(path, columns=MATCH_COLUMNS)
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
    if pl.scan_parquet(gp_dir / '*.parquet').select(pl.len()).collect().item() == 0:
        return cd.with_columns(genes=pl.lit(None, dtype=pl.List(pl.String))).collect()
    keys = cd.select('pub', 'h').unique()
    gp = (
        pl
        .scan_parquet(gp_dir / '*.parquet')
        .join(keys, on=['pub', 'h'], how='semi')
        .group_by('pub', 'h')
        .agg(genes=pl.col('gene').unique().sort())
    )
    return cd.join(gp, on=['pub', 'h'], how='left').collect(engine='streaming')


def select_sentences(sentences: pl.DataFrame) -> pl.DataFrame:
    """Pick the sentences that represent each drug, deterministically.

    Texts are stripped and kept when their length lies between ``MIN_CHARS`` and ``MAX_CHARS``.
    A text repeated within a drug keeps its best row (cue, then gene, then section rank, newer
    year, then ``pub``, ``h`` and ``label``). Sentences are tiered (0 cue and gene, 1 cue,
    2 gene, 3 neither), ranked inside each tier and spread across publications, and the first
    ``PER_DRUG`` per drug are kept. The result does not depend on the input row order.

    Args:
        sentences: Output of ``join_genes``.

    Returns:
        The input columns plus ``gene``, ``tier``, ``nth_in_pub`` and ``doc`` (``pmid<id>``, else the
        lowercased PMC id), ordered by drug and then by selection rank.
    """
    deduped = (
        sentences
        .with_columns(gene=pl.col('genes').is_not_null(), text=pl.col('text').str.strip_chars())
        .filter(pl.col('text').str.len_chars().is_between(MIN_CHARS, MAX_CHARS))
        .sort(
            'parent',
            'text',
            'cue',
            'gene',
            'srank',
            'year',
            'pub',
            'h',
            'label',
            'mappedId',
            'section',
            'pmid',
            'pmcid',
            descending=[False, False, True, True, False, True, False, False, False, False, False, False, False],
        )
        .unique(['parent', 'text'], keep='first', maintain_order=True)
    )
    tier = (
        pl
        .when(pl.col('cue') & pl.col('gene'))
        .then(0)
        .when(pl.col('cue'))
        .then(1)
        .when(pl.col('gene'))
        .then(2)
        .otherwise(3)
    )
    spread = ['parent', 'tier', 'nth_in_pub', 'srank', 'year', 'pub', 'h']
    return (
        deduped
        .with_columns(tier=tier)
        .sort(
            'parent',
            'tier',
            'srank',
            'year',
            'h',
            'text',
            descending=[False, False, False, True, False, False],
            maintain_order=True,
        )
        .with_columns(nth_in_pub=pl.int_range(pl.len()).over('parent', 'pub', 'tier'))
        .sort(spread, descending=[False, False, False, False, True, False, False], maintain_order=True)
        .filter(pl.int_range(pl.len()).over('parent') < PER_DRUG)
        .with_columns(
            doc=pl
            .when(pl.col('pmid').is_not_null())
            .then('pmid' + pl.col('pmid'))
            .otherwise(pl.col('pmcid').str.to_lowercase())
        )
    )


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def content_key(parent: str, sentences: list[dict[str, Any]]) -> str:
    """Key a drug's request by its selected sentences alone.

    Args:
        parent: Parent molecule id.
        sentences: Dicts with ``doc`` and ``sha256``, in any order.

    Returns:
        ``parent|`` followed by the SHA-256 hex digest of ``json.dumps`` (default separators) of
        the sorted ``[doc, sha256]`` pairs.
    """
    pairs = sorted([s['doc'], s['sha256']] for s in sentences)
    return f'{parent}|{_sha256(json.dumps(pairs))}'


def requests(picked: pl.DataFrame, names: pl.DataFrame) -> pl.DataFrame:
    """Build the extraction request of each drug from its selected sentences.

    Within a document the sentences keep the order of ``picked`` after a stable sort by section
    rank. Documents are ordered by id. Documents without a publication year or an id are left out of the
    request text, ``sentences``, ``docs`` and the content key (their drug names still appear in the
    names line). Drugs with no remaining document, and drugs whose parent is missing from
    ``names``, get no row.

    Args:
        picked: Output of ``select_sentences``.
        names: Frame with ``parent`` and ``name``.

    Returns:
        Frame with ``parent``, ``name``, ``request``, ``content_key``, ``sentences`` (list of
        ``{doc, sha256, cue}`` sorted by ``(doc, sha256)``) and ``docs`` (list of ``{doc, text}``,
        the drug's sentences from that document joined by newline), ordered by ``parent``.
    """
    docs = (
        picked
        .sort('parent', 'doc', 'srank', maintain_order=True)
        .group_by('parent', 'doc', maintain_order=True)
        .agg(pl.col('year').first(), sentences=pl.col('text'), labels=pl.col('label').unique())
        .with_columns(
            text=pl.col('sentences').list.join('\n'),
            prompt='Trial ID: '
            + pl.col('doc')
            + '\nPublication year: '
            + pl.col('year').cast(pl.String)
            + '\nSentences:\n'
            + pl.col('sentences').list.join('\n'),
        )
    )
    shown = pl.col('year').is_not_null() & pl.col('doc').is_not_null()
    drugs = (
        docs
        .group_by('parent', maintain_order=True)
        .agg(
            text=pl.col('prompt').filter(shown).str.join('\n---\n'),
            labels=pl.col('labels').list.explode(keep_nulls=False, empty_as_null=False).unique().sort(),
            docs=pl.struct('doc', 'text').filter(shown),
        )
        .filter(pl.col('docs').list.len() > 0)
    )
    named = drugs.join(names.select('parent', name=pl.col('name').str.to_lowercase()), on='parent')
    if named.height < drugs.height:
        logger.info(f'{drugs.height - named.height} drugs without a name get no request')
    sentences = (
        picked
        .join(docs.filter(shown).select('parent', 'doc'), on=['parent', 'doc'], how='semi')
        .select(
            'parent',
            'doc',
            'cue',
            sha256=pl.col('text').map_elements(_sha256, return_dtype=pl.String),
        )
        .sort('parent', 'doc', 'sha256')
        .group_by('parent', maintain_order=True)
        .agg(sentences=pl.struct('doc', 'sha256', 'cue'))
    )
    out = (
        named
        .join(sentences, on='parent')
        .with_columns(
            request='Drug: '
            + pl.col('name')
            + '\nNames of the drug in these sentences: '
            + pl.col('labels').list.join(', ')
            + '\n\n'
            + pl.col('text')
        )
        .sort('parent')
    )
    keys = [content_key(p, s) for p, s in zip(out['parent'], out['sentences'].to_list(), strict=True)]
    return out.with_columns(content_key=pl.Series(keys, dtype=pl.String)).select(
        'parent', 'name', 'request', 'content_key', 'sentences', 'docs'
    )
