"""Reusable JSON dump for pandera.polars DataFrameModel schemas.

pandera's own `DataFrameSchema.to_json()` only round-trips checks it can describe declaratively --
see `Check.is_builtin_check` -- so a hand-written check (a Python function, like the ones in
`pts.schemas.evidence`) is silently skipped with a warning: there is no generic way to serialize
arbitrary code as data. The check itself is unaffected -- it still runs at `validate()` time --
only its description in this JSON dump was ever in question. It also knows nothing about the
`metadata=` dict `pa.Field`/`Config` carry, since that is a PTS convention (`foreign_key`,
`primary_key`, `bioregistry`, ...), not a pandera concept.

`schema_to_dict` fills both gaps by augmenting `to_json()`'s own output: it adds each
column's/dataframe's `get_metadata()` metadata, and a `custom_checks` list (name + docstring) for
every check `to_json()` had to drop, so nothing about the schema goes undocumented in the dump.
"""

import inspect
import json
import warnings
from typing import Any

import pandera.polars as pa
from pandera.api.checks import Check


def schema_to_dict(model: type[pa.DataFrameModel]) -> dict[str, Any]:
    """Serialize a pandera.polars `DataFrameModel` to a JSON-ready dict.

    Args:
        model: a `pandera.polars.DataFrameModel` subclass (the schema class itself, not an
            instance -- e.g. `GwasCredibleSetEvidenceSchema`, not `GwasCredibleSetEvidenceSchema()`).

    Returns:
        The dict `model.to_schema().to_json()` would produce, plus a `metadata` key (from
        `model.get_metadata()`) on the dataframe and on each column that has one, and a
        `custom_checks` key (list of `{name, description}`) on the dataframe and on each column
        that has a check `to_json()` couldn't describe declaratively.
    """
    schema = model.to_schema()
    with warnings.catch_warnings():
        # expected: `custom_checks` below is exactly how we cover what `to_json()` had to skip.
        warnings.filterwarnings('ignore', message='Only registered checks may be serialized')
        dumped: dict[str, Any] = json.loads(schema.to_json())
    metadata_by_name = model.get_metadata()
    if metadata_by_name is None:
        msg = f'{model.__name__}.get_metadata() returned None -- expected a dict keyed by schema name'
        raise TypeError(msg)
    metadata = metadata_by_name[schema.name]

    for column_name, column_schema in schema.columns.items():
        column_dump = dumped['columns'][column_name]
        column_metadata = metadata['columns'].get(column_name)
        if column_metadata:
            column_dump['metadata'] = column_metadata
        custom_checks = _describe_custom_checks(column_schema.checks, model)
        if custom_checks:
            column_dump['custom_checks'] = custom_checks

    if metadata.get('dataframe'):
        dumped['metadata'] = metadata['dataframe']
    dataframe_custom_checks = _describe_custom_checks(schema.checks, model)
    if dataframe_custom_checks:
        dumped['custom_checks'] = dataframe_custom_checks

    return dumped


def _describe_custom_checks(checks: list[Check] | None, model: type[pa.DataFrameModel]) -> list[dict[str, str]]:
    """Name + docstring for every check in `checks` that `to_json()` can't describe declaratively."""
    described: list[dict[str, str]] = []
    for check in checks or []:
        name = check.name
        if name is None or Check.is_builtin_check(name):
            continue
        described.append({'name': name, 'description': inspect.getdoc(getattr(model, name)) or ''})
    return described
