"""Colocated variants matched as VEP 116's `--check_existing --exclude_null_alleles` matches them."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import cast

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.vep import vep_schema
from pts.transformers.variant.alleles import shared_prefix_length, vep_allele_form

KEY = ['chromosome', 'position', 'referenceAllele', 'alternateAllele']
CACHE_COLUMNS = {
    'chromosome': 0,
    'variation_name': 1,
    'failed': 2,
    'start': 4,
    'end': 5,
    'allele_string': 6,
    'strand': 7,
    'var_synonyms': 14,
}
"""The `all_vars` columns read, by their zero-based index in the cache's `variation_cols`."""

_COLOCATED = cast(pl.List, vep_schema['colocated_variants'])
_SYNONYMS = cast(pl.Struct, cast(pl.Struct, _COLOCATED.inner).to_schema()['var_synonyms'])


def extract_all_vars(tarball: str, directory: Path) -> dict[str, Path]:
    """Each chromosome's `all_vars.gz` from the packed indexed VEP cache, in one streaming pass.

    Members `*/<chromosome>/all_vars.gz` are written to `directory/<chromosome>.all_vars.gz`;
    everything else in the tarball is skipped.

    Raises:
        ValueError: if the tarball holds no `all_vars.gz`.
    """
    directory.mkdir(parents=True, exist_ok=True)
    extracted: dict[str, Path] = {}
    with StorageHandle(tarball).open('rb') as raw, tarfile.open(fileobj=raw, mode='r|gz') as archive:
        for member in archive:
            parts = member.name.split('/')
            if not member.isfile() or len(parts) < 3 or parts[-1] != 'all_vars.gz':
                continue
            source = archive.extractfile(member)
            if source is None:
                continue
            path = directory / f'{parts[-2]}.all_vars.gz'
            with source, path.open('wb') as target:
                shutil.copyfileobj(source, target)
            extracted[parts[-2]] = path
    if not extracted:
        raise ValueError(f'{tarball}: no all_vars.gz')
    return extracted


def _cache(all_vars: Path, positions: pl.DataFrame) -> pl.DataFrame:
    """The usable `all_vars` lines whose start is one of `positions`, with their line number.

    Failed variants and allele strings without `/` are dropped; `.` is null, a null end is the
    start and a null strand is forward.
    """
    return (
        pl
        .scan_csv(
            all_vars,
            separator='\t',
            has_header=False,
            quote_char=None,
            comment_prefix='#',
            infer_schema=False,
            null_values='.',
            truncate_ragged_lines=True,
        )
        .select([pl.nth(index).alias(name) for name, index in CACHE_COLUMNS.items()])
        .with_row_index('line')
        .filter(
            (pl.col('failed').cast(pl.Int64, strict=False).fill_null(0) <= 0)
            & pl.col('allele_string').str.contains('/', literal=True)
        )
        .with_columns(pl.col('start').cast(pl.Int64))
        .join(positions.lazy(), on=['chromosome', 'start'], how='semi')
        .collect(engine='streaming')
        .with_columns(
            pl.coalesce(pl.col('end').cast(pl.Int64, strict=False), 'start').alias('end'),
            pl.col('strand').cast(pl.Int64, strict=False).fill_null(1).alias('strand'),
        )
    )


def _trimmed(frame: pl.DataFrame, end_first: bool) -> pl.DataFrame:
    """`trim_sequences` of each (`ref`, `alt`, `pos`) as a `key`, empty alleles written `-`.

    Left-first trims the shared prefix, advancing `pos`, then the shared suffix; end-first
    trims the suffix, then the prefix.
    """
    ref, alt = pl.col('_ref'), pl.col('_alt')
    frame = frame.with_columns(pl.col('ref').alias('_ref'), pl.col('alt').alias('_alt'), pl.lit(0).alias('_prefix'))
    steps = ('suffix', 'prefix') if end_first else ('prefix', 'suffix')
    for step in steps:
        if step == 'prefix':
            frame = frame.with_columns(shared_prefix_length(ref, alt).alias('_prefix')).with_columns(
                ref.str.slice(pl.col('_prefix')), alt.str.slice(pl.col('_prefix'))
            )
        else:
            frame = frame.with_columns(
                shared_prefix_length(ref.str.reverse(), alt.str.reverse()).alias('_suffix')
            ).with_columns(
                ref.str.head(ref.str.len_chars() - pl.col('_suffix')),
                alt.str.head(alt.str.len_chars() - pl.col('_suffix')),
            )
    return frame.select(
        'pair',
        pl.concat_str(
            pl.when(ref == '').then(pl.lit('-')).otherwise(ref),
            pl.when(alt == '').then(pl.lit('-')).otherwise(alt),
            (pl.col('pos') + pl.col('_prefix')).cast(pl.String),
            separator='_',
        ).alias('key'),
    )


def _keys(alleles: pl.DataFrame) -> pl.DataFrame:
    """The (`pair`, `key`) of each ref/alt pair: left-first, and end-first when an allele is longer than one base."""
    longer = alleles.filter((pl.col('ref').str.len_chars() > 1) | (pl.col('alt').str.len_chars() > 1))
    return pl.concat([_trimmed(alleles, end_first=False), _trimmed(longer, end_first=True)]).unique()


def _reverse_complement(allele: pl.Expr) -> pl.Expr:
    return allele.str.reverse().str.replace_many(['A', 'C', 'G', 'T'], ['T', 'G', 'C', 'A'])


def _synonyms(source: str) -> pl.Expr:
    """The `source` identifiers in a `var_synonyms` cell (`ClinVar::RCV1,VCV2--OMIM::171834.0001`)."""
    found = (
        pl
        .col('var_synonyms')
        .str.split('--')
        .list.eval(
            pl
            .element()
            .filter(pl.element().str.starts_with(f'{source}::'))
            .str.strip_prefix(f'{source}::')
            .str.split(',')
        )
        .list.eval(pl.element().explode())
    )
    return pl.when(found.list.len() > 0).then(found)


def colocated_variants(all_vars: Path, rows: pl.DataFrame) -> pl.DataFrame:
    """The known variants VEP 116 reports as colocated with each row, from the cache's `all_vars.gz`.

    Each row is taken in VEP's input form (`vep_allele_form`). Candidates are the cached
    variants whose start lies in [start - 1, end + 1], an insertion ending before it starts;
    failed variants and null allele strings are dropped. Both sides' ref/alt pairs are
    trimmed as `get_matched_variant_alleles` trims them, the input reverse-complemented against
    a reverse-strand cached variant, and a candidate matches on any shared (ref, alt, pos).
    Each id is kept once, in cache line order.

    Returns:
        `chromosome`, `position`, `referenceAllele`, `alternateAllele` and `colocated_variants`
        in VEP's shape, for the distinct rows with at least one colocated variant, in row order.
    """
    variants = vep_allele_form(rows.lazy().select(KEY).unique(maintain_order=True)).with_row_index('variant').collect()
    windows = variants.select(
        'variant',
        'chromosome',
        pl.int_ranges(pl.col('vepStart') - 1, pl.col('vepEnd') + 2, dtype=pl.Int64).alias('start'),
    ).explode('start')
    cache = _cache(all_vars, windows.select('chromosome', 'start').unique())
    pairs = (
        windows
        .join(cache, on=['chromosome', 'start'])
        .join(
            variants.select('variant', 'vepStart', 'vepReference', 'vepAlternate'),
            on='variant',
            how='left',
            maintain_order='left',
        )
        .with_row_index('pair')
    )
    reverse = pl.col('strand') == -1
    inputs = pairs.select(
        'pair',
        pl.when(reverse).then(_reverse_complement(pl.col('vepReference'))).otherwise('vepReference').alias('ref'),
        pl.when(reverse).then(_reverse_complement(pl.col('vepAlternate'))).otherwise('vepAlternate').alias('alt'),
        pl.col('vepStart').alias('pos'),
    )
    alleles = pl.col('allele_string').str.strip_chars_end('/').str.split('/')
    cached = pairs.select(
        'pair', alleles.list.first().alias('ref'), alleles.list.slice(1).alias('alt'), pl.col('start').alias('pos')
    ).explode('alt')
    matched = _keys(inputs).join(_keys(cached), on=['pair', 'key'], how='semi').select('pair').unique()
    colocated = (
        pairs
        .join(matched, on='pair', how='semi')
        .sort('variant', 'line')
        .unique(['variant', 'variation_name'], keep='first', maintain_order=True)
        .rechunk()
        .group_by('variant', maintain_order=True)
        .agg(
            pl.struct(
                pl.col('variation_name').alias('id'),
                'allele_string',
                'start',
                'end',
                pl
                .when(pl.col('var_synonyms').is_not_null())
                .then(pl.struct(_synonyms('ClinVar').alias('ClinVar'), _synonyms('OMIM').alias('OMIM')))
                .cast(_SYNONYMS)
                .alias('var_synonyms'),
            ).alias('colocated_variants')
        )
    )
    return variants.join(colocated, on='variant', how='inner', maintain_order='left').select(
        *KEY, pl.col('colocated_variants').cast(_COLOCATED)
    )
