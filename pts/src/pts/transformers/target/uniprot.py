"""UniProt entries: synonyms, functions, cross-references and subcellular locations."""

from __future__ import annotations

import gzip
import re
from typing import IO

import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.target import UNIPROT_ENTRY
from pts.transformers.target.lists import tag, union
from pts.transformers.utils.dataset import scan_dataset

_DB_INTEREST = {'ChEMBL', 'DrugBank', 'PDB', 'Ensembl', 'GO', 'InterPro', 'Reactome'}

_BRACES_RE = re.compile(r'\{[^}]*\}')
_MODIFIER_RE = re.compile(r'^\[([^\]]+)\]:\s*(.+)$')


def _strip_braces(text: str) -> str:
    return _BRACES_RE.sub('', text).strip()


def _parse_record(lines: list[str]) -> dict:
    entry_id = ''
    accessions: list[str] = []
    names: list[str] = []
    synonyms: list[str] = []
    symbol_synonyms: list[str] = []
    db_xrefs: list[str] = []
    functions: list[str] = []
    locations: list[dict] = []
    gn_lines: list[str] = []  # accumulated across continuation lines

    # CC state machine
    cc_topic: str | None = None
    cc_lines: list[str] = []

    def _flush_cc() -> None:
        if cc_topic is None:
            return
        text = ' '.join(cc_lines).strip()
        if cc_topic == 'FUNCTION':
            if text:
                functions.append(text)
        elif cc_topic == 'SUBCELLULAR LOCATION':
            text = _strip_braces(text)
            # Split by period; stop at first Note= segment (its continuation
            # sentences don't start with Note= but are still part of the note)
            parts = text.split('.')
            for part in parts:
                p = part.strip()
                if not p:
                    continue
                if p.startswith(('Note=', 'Note ')):
                    break
                m = _MODIFIER_RE.match(p)
                if m:
                    locations.append({
                        'location': m.group(2).strip(),
                        'targetModifier': m.group(1).strip(),
                    })
                else:
                    locations.append({'location': p, 'targetModifier': None})

    for raw in lines:
        if len(raw) < 2:
            continue
        code = raw[:2]
        content = raw[5:] if len(raw) > 5 else ''

        if code == 'ID':
            tokens = content.split()
            if tokens:
                entry_id = tokens[0]

        elif code == 'AC':
            content_clean = _strip_braces(content)
            for ac in content_clean.split(';'):
                ac = ac.strip()
                if ac:
                    accessions.append(ac)

        elif code == 'DE':
            content_clean = _strip_braces(content).strip()
            if content_clean.startswith('RecName: Full='):
                val = content_clean[len('RecName: Full=') :].rstrip(';').strip()
                if val:
                    names.append(val)
            elif content_clean.startswith('AltName: Full='):
                val = content_clean[len('AltName: Full=') :].rstrip(';').strip()
                if val:
                    synonyms.append(val)
            elif content_clean.startswith('AltName: CD_antigen='):
                val = content_clean[len('AltName: CD_antigen=') :].rstrip(';').strip()
                if val:
                    symbol_synonyms.append(val)
            elif content_clean.startswith('Short='):
                val = content_clean[len('Short=') :].rstrip(';').strip()
                if val:
                    symbol_synonyms.append(val)

        elif code == 'GN':
            gn_lines.append(content)

        elif code == 'DR':
            content_clean = _strip_braces(content).strip()
            tokens = [t.rstrip(';').rstrip('.').strip() for t in content_clean.split(';')]
            tokens = [t for t in tokens if t]
            if tokens and tokens[0] in _DB_INTEREST:
                if len(tokens) >= 2:
                    db_xrefs.append(f'{tokens[1]} {tokens[0]}')

        elif code == 'CC':
            content_clean = _strip_braces(content)
            if content_clean.lstrip().startswith('-----'):
                # Copyright separator — flush and stop CC parsing
                _flush_cc()
                cc_topic = None
                cc_lines = []
            elif content_clean.startswith('-!- '):
                # Flush previous CC block
                _flush_cc()
                cc_lines = []
                # Extract topic and first line of content
                rest = content_clean[4:]
                if ': ' in rest:
                    topic, _, first_line = rest.partition(': ')
                    cc_topic = topic.strip()
                    first_line = first_line.strip()
                    if first_line:
                        cc_lines.append(first_line)
                else:
                    cc_topic = rest.strip()
            elif cc_topic is not None:
                # Continuation line — strip leading spaces
                continuation = content_clean.strip()
                if continuation:
                    cc_lines.append(continuation)

    _flush_cc()

    # Parse accumulated GN lines as one block so _strip_braces handles
    # multi-line {ECO:...} evidence codes correctly.
    gn_text = _strip_braces(' '.join(gn_lines))
    for segment in gn_text.split(';'):
        segment = segment.strip()
        if not segment:
            continue
        if '=' in segment:
            key, _, value = segment.partition('=')
            key = key.strip()
            value = value.strip()
            if key in ('Name', 'Synonyms', 'ORFNames'):
                for v in value.split(','):
                    v = v.strip().rstrip(';')
                    if v:
                        symbol_synonyms.append(v)

    return {
        'id': entry_id,
        'accessions': accessions,
        'names': names,
        'synonyms': synonyms,
        'symbolSynonyms': symbol_synonyms,
        'dbXrefs': db_xrefs,
        'functions': functions,
        'locations': locations,
    }


def _parse_uniprot(file_obj: IO[bytes]) -> list[dict]:
    records: list[dict] = []
    current_lines: list[str] = []
    count = 0

    for raw_line in file_obj:
        if isinstance(raw_line, bytes):
            line = raw_line.decode('utf-8', errors='replace').rstrip('\n')
        else:
            line = raw_line.rstrip('\n')

        if line.startswith('//'):
            if current_lines:
                records.append(_parse_record(current_lines))
                count += 1
                if count % 10000 == 0:
                    logger.info(f'parsed {count} uniprot records')
                current_lines = []
        else:
            current_lines.append(line)

    # Handle any trailing record without '//'
    if current_lines:
        records.append(_parse_record(current_lines))

    logger.info(f'parsed {len(records)} uniprot records total')
    return records


def read_uniprot(path: str) -> pl.DataFrame:
    """Parse the UniProt flat file, optionally gzipped, one row per entry."""
    logger.info(f'loading uniprot flat file from {path}')
    with StorageHandle(path).open('rb') as raw:
        if path.endswith('.gz'):
            with gzip.open(raw, 'rb') as gz:
                rows = _parse_uniprot(gz)
        else:
            rows = _parse_uniprot(raw)
    return pl.DataFrame(rows, schema=UNIPROT_ENTRY)


def read_uniprot_ssl(path: str) -> pl.DataFrame:
    """The UniProt subcellular location vocabulary, every column as text."""
    return scan_dataset(path, format='tsv', has_header=True, quote_char=None, infer_schema=False).collect()


#: Java's `\w` and `\s`, which are ASCII; Rust's are unicode.
_WORD_OR_SPACE = r'[A-Za-z0-9_ \t\n\x0B\f\r]'
_SPACE = r'[ \t\n\x0B\f\r]'
#: Java's `.`, which also stops at these line terminators.
_ANY = r'[^\n\r\x{85}\x{2028}\x{2029}]'


def _ssl_match() -> pl.Expr:
    first_words = pl.col('location').str.extract(rf'^({_WORD_OR_SPACE}+)', 1).fill_null('').str.strip_chars(' ')
    after_comma = (
        pl.col('location').str.extract(rf'{_ANY}*,{_SPACE}({_WORD_OR_SPACE}+)', 1).fill_null('').str.strip_chars(' ')
    )
    return pl.when(first_words != '').then(first_words).when(after_comma != '').then(after_comma)


def subcellular_locations(entries: pl.DataFrame, ssl: pl.DataFrame) -> pl.DataFrame:
    """UniProt locations mapped to subcellular location terms, per entry, in entry order.

    Returns:
        [uniprotId, subcellularLocations]; entries without locations are absent.
    """
    vocabulary = ssl.select(
        pl.col('Subcellular location ID').alias('termSL'),
        pl.col('Name').alias('ssl_match'),
        pl.col('Category').alias('labelSL'),
    )
    return (
        entries
        .select('uniprotId', 'locations')
        .explode('locations', empty_as_null=False, keep_nulls=False)
        .unnest('locations')
        .filter(pl.col('location').is_not_null())
        .with_columns(_ssl_match().alias('ssl_match'))
        .join(vocabulary, on='ssl_match', how='left', maintain_order='left')
        .with_columns(pl.lit('uniprot').alias('source'))
        .group_by('uniprotId', maintain_order=True)
        .agg(pl.struct('location', 'source', 'termSL', 'labelSL', 'targetModifier').alias('subcellularLocations'))
    )


def build_uniprot(entries: pl.DataFrame, ssl: pl.DataFrame) -> pl.DataFrame:
    """One row per UniProt entry, keyed by its primary accession.

    Returns:
        [uniprotId, functionDescriptions, dbXrefs, subcellularLocations, proteinIds, synonyms,
        symbolSynonyms, nameSynonyms].
    """
    base = entries.filter(pl.col('accessions').list.len() > 0).select(
        pl.col('accessions').list.first().alias('uniprotId'),
        union(pl.col('names'), pl.col('synonyms')).alias('nameSynonyms'),
        union(pl.col('symbolSynonyms')).alias('symbolSynonyms'),
        union(pl.col('names'), pl.col('symbolSynonyms')).alias('synonyms'),
        pl.col('functions').alias('functionDescriptions'),
        pl
        .col('dbXrefs')
        .list.eval(
            pl.struct(
                pl.element().str.split(' ').list.get(0, null_on_oob=True).alias('id'),
                pl.element().str.split(' ').list.get(1, null_on_oob=True).alias('source'),
            )
        )
        .alias('dbXrefs'),
        pl
        .col('accessions')
        .list.eval(pl.struct(pl.element().str.strip_chars(' ').alias('id'), pl.lit('uniprot_obsolete').alias('source')))
        .alias('proteinIds'),
        'locations',
    )
    return (
        base
        .drop('locations')
        .join(subcellular_locations(base, ssl), on='uniprotId', how='left', maintain_order='left')
        .select(
            'uniprotId',
            'functionDescriptions',
            'dbXrefs',
            'subcellularLocations',
            'proteinIds',
            tag(pl.col('synonyms'), 'label', 'uniprot').alias('synonyms'),
            tag(pl.col('symbolSynonyms'), 'label', 'uniprot').alias('symbolSynonyms'),
            tag(pl.col('nameSynonyms'), 'label', 'uniprot').alias('nameSynonyms'),
        )
    )
