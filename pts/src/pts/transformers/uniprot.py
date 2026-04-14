from __future__ import annotations

import gzip
import re
from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

_UNIPROT_SCHEMA = pl.Schema({
    'id': pl.String,
    'accessions': pl.List(pl.String),
    'names': pl.List(pl.String),
    'synonyms': pl.List(pl.String),
    'symbolSynonyms': pl.List(pl.String),
    'dbXrefs': pl.List(pl.String),
    'functions': pl.List(pl.String),
    'locations': pl.List(pl.String),
})

_DB_INTEREST = {'ChEMBL', 'DrugBank', 'PDB', 'Ensembl', 'GO', 'InterPro', 'Reactome'}

_BRACES_RE = re.compile(r'\{[^}]*\}')


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
    locations: list[str] = []
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
                locations.append(p)

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


def _parse_uniprot(file_obj) -> list[dict]:
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


def uniprot(
    source: str,
    destination: str,
    settings: dict[str, Any],
    config: Config,
) -> None:
    logger.info(f'loading uniprot flat file from {source}')
    h = StorageHandle(source)

    with h.open('rb') as raw:
        if source.endswith('.gz'):
            with gzip.open(raw, 'rb') as gz:
                rows = _parse_uniprot(gz)
        else:
            rows = _parse_uniprot(raw)

    logger.info('creating dataframe from parsed uniprot data')
    df = pl.DataFrame(rows, schema=_UNIPROT_SCHEMA)

    logger.info(f'writing uniprot parquet to {destination}')
    df.write_parquet(destination)
