from __future__ import annotations

from typing import Any, TextIO, cast

import obonet
import polars as pl
from loguru import logger
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.go import go_schema


def _rows_from_obo(graph) -> list[dict]:
    rows: list[dict] = []
    obsolete_count = 0
    for go_id, data in graph.nodes(data=True):
        if not go_id.startswith('GO:'):
            continue
        label = data.get('name')
        namespace = data.get('namespace')
        alt_ids = data.get('alt_id') or []
        # robust parse: obonet stores 'is_obsolete' as 'true' (str) when present
        obs_value = data.get('is_obsolete', False)
        if isinstance(obs_value, str):
            is_obsolete = obs_value.lower() == 'true'
        else:
            is_obsolete = bool(obs_value)
        if is_obsolete:
            obsolete_count += 1
            if obsolete_count <= 3:
                logger.debug(
                    f'obsolete term found: {go_id}, raw value: {obs_value!r}, type: {type(obs_value).__name__}'
                )
        is_a = data.get('is_a') or []
        # relationships may come as strings 'rel TARGET' or tuples ('rel', 'TARGET')
        relationships = data.get('relationship') or []
        part_of: list[str] = []
        regulates: list[str] = []
        negatively_regulates: list[str] = []
        positively_regulates: list[str] = []

        for rel in relationships:
            if isinstance(rel, tuple) and len(rel) >= 2:
                key, target = rel[0], rel[1]
            elif isinstance(rel, str) and ' ' in rel:
                key, target = rel.split(' ', 1)
            else:
                continue

            if key == 'part_of':
                part_of.append(target)
            elif key == 'regulates':
                regulates.append(target)
            elif key == 'negatively_regulates':
                negatively_regulates.append(target)
            elif key == 'positively_regulates':
                positively_regulates.append(target)

        rows.append({
            'id': go_id,
            'label': label,
            'namespace': namespace,
            'altIds': alt_ids,
            'isA': is_a,
            'partOf': part_of,
            'regulates': regulates,
            'negativelyRegulates': negatively_regulates,
            'positivelyRegulates': positively_regulates,
            'isObsolete': is_obsolete,
        })
    logger.info(f'parsed {len(rows)} go terms, {obsolete_count} obsolete')
    return rows


def go(source: str, destination: str, settings: dict[str, Any]) -> None:
    logger.info('loading go obo file')
    h = StorageHandle(source)
    f = h.open('rt')
    graph = obonet.read_obo(cast(TextIO, f), ignore_obsolete=False)

    logger.info('transforming obo data')
    rows = _rows_from_obo(graph)

    logger.info('creating dataframe from parsed data')
    df = pl.DataFrame(rows, schema=go_schema)

    logger.info('writing dataframe')
    df.write_parquet(destination)
