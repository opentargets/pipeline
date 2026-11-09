"""Colocated variants matched as VEP 116's `--check_existing --exclude_null_alleles` matches them."""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path

import polars as pl
from otter.storage.synchronous.handle import StorageHandle

from pts.transformers.variant.vep.alleles import shared_prefix_length, vep_allele_form
from pts.transformers.variant.vep.frames import ID_LISTS

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


def _trimmed(alleles: pl.DataFrame, end_first: bool) -> pl.DataFrame:
    """`trim_sequences` of each (`ref`, `alt`): `_alleles`, the trimmed pair, and `_prefix`, how far `pos` advances.

    Left-first trims the shared prefix, then the shared suffix; end-first trims the suffix,
    then the prefix. Empty alleles are written `-`, and `_alleles` joins the two with `_`.
    """
    ref, alt = pl.col('_ref'), pl.col('_alt')
    frame = alleles.with_columns(pl.col('ref').alias('_ref'), pl.col('alt').alias('_alt'), pl.lit(0).alias('_prefix'))
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
        'ref',
        'alt',
        pl.concat_str(
            pl.when(ref == '').then(pl.lit('-')).otherwise(ref),
            pl.when(alt == '').then(pl.lit('-')).otherwise(alt),
            separator='_',
        ).alias('_alleles'),
        '_prefix',
    )


def _keys(alleles: pl.DataFrame) -> pl.DataFrame:
    """The (`pair`, `key`) of each (`ref`, `alt`, `pos`), trimmed as `_trimmed` trims them.

    Every pair is trimmed left-first, and end-first too when an allele is longer than one base.
    Each distinct (`ref`, `alt`) is trimmed once.
    """
    distinct = alleles.select('ref', 'alt').unique()
    longer = distinct.filter((pl.col('ref').str.len_chars() > 1) | (pl.col('alt').str.len_chars() > 1))
    trims = pl.concat([_trimmed(distinct, end_first=False), _trimmed(longer, end_first=True)])
    return (
        alleles
        .join(trims, on=['ref', 'alt'])
        .select(
            'pair',
            pl.concat_str('_alleles', (pl.col('pos') + pl.col('_prefix')).cast(pl.String), separator='_').alias('key'),
        )
        .unique()
    )


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


def omim_ids(colocated: pl.Expr) -> pl.Expr:
    """OMIM cross-references, written `<entry>#<variant>` as on omim.org.

    The variant suffix is padded to its four digits, which VEP's numbers do not always keep
    (`600234.001` -> `600234#0010`).
    """
    synonyms = colocated.list.eval(
        pl.element().struct.field('var_synonyms').struct.field('OMIM').drop_nulls().explode()
    )
    parts = pl.element().str.split_exact('.', 1)
    entry, suffix = parts.struct.field('field_0'), parts.struct.field('field_1')
    return synonyms.list.eval(
        pl.when(suffix.is_null()).then(entry).otherwise(pl.concat_str(entry, pl.lit('#'), suffix.str.pad_end(4, '0')))
    )


def clinvar_ids(colocated: pl.Expr) -> pl.Expr:
    """ClinVar variation (VCV) identifiers, each once, in first-occurrence order."""
    clinvar = colocated.list.eval(
        pl.element().struct.field('var_synonyms').struct.field('ClinVar').drop_nulls().explode()
    )
    return clinvar.list.eval(pl.element().filter(pl.element().str.starts_with('VCV'))).list.unique(maintain_order=True)


def colocated_variants(all_vars: Path, rows: pl.DataFrame) -> pl.DataFrame:
    """The known variants VEP 116 reports as colocated with each row, from the cache's `all_vars.gz`.

    Each row is taken in VEP's input form (`vep_allele_form`). Candidates are the cached
    variants whose start lies in [start - 1, end + 1], an insertion ending before it starts;
    failed variants and null allele strings are dropped. Both sides' ref/alt pairs are
    trimmed as `get_matched_variant_alleles` trims them, the input reverse-complemented against
    a reverse-strand cached variant, and a candidate matches on any shared (ref, alt, pos).
    Each id is kept once, in cache line order.

    Returns:
        `chromosome`, `position`, `referenceAllele`, `alternateAllele` and the `ID_LISTS` of
        the colocated variants (`omim_ids`, `clinvar_ids`), for the distinct rows with at least
        one colocated variant, in row order.
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
                pl
                .when(pl.col('var_synonyms').is_not_null())
                .then(pl.struct(_synonyms('ClinVar').alias('ClinVar'), _synonyms('OMIM').alias('OMIM')))
                .alias('var_synonyms'),
            ).alias('colocated')
        )
    )
    found = pl.col('colocated')
    return variants.join(colocated, on='variant', how='inner', maintain_order='left').select(
        *KEY,
        found.list.eval(pl.element().struct.field('id')).alias('rsIds'),
        clinvar_ids(found).alias('clinvarIds'),
        omim_ids(found).alias('omimIds'),
    )


def add_colocated(variants: pl.DataFrame, colocated: pl.DataFrame | None) -> pl.DataFrame:
    """The variants with the `ID_LISTS` of their colocated variants, `[]` for none or without `colocated`."""
    if colocated is None:
        return variants.with_columns(pl.lit([], dtype).alias(name) for name, dtype in ID_LISTS.items())
    return variants.join(colocated.select(*KEY, *ID_LISTS), on=KEY, how='left', maintain_order='left').with_columns(
        pl.col(name).fill_null(pl.lit([], dtype)) for name, dtype in ID_LISTS.items()
    )
