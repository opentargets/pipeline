"""Reusable JSON dump for pandera.polars DataFrameModel schemas.

pandera's own `DataFrameSchema.to_json()` has a few gaps for schemas built the way this project
builds them:

* It only round-trips checks it can describe declaratively -- see `Check.is_builtin_check` -- so a
  hand-written check (a Python function, like the ones in `pts.schemas.evidence`) is silently
  skipped with a warning: there is no generic way to serialize arbitrary code as data. The check
  itself is unaffected -- it still runs at `validate()` time -- only its description in this JSON
  dump was ever in question.
* It knows nothing about the `metadata=` dict `pa.Field`/`Config` carry, since that is a PTS
  convention (`foreign_key`, `primary_key`, `bioregistry`, ...), not a pandera concept.
* It only writes `"nullable": true` for a column that IS nullable, and omits the key entirely for
  one that isn't -- `nullable=False` (pandera's default, actively enforced -- see
  `not_nullable`/`SERIES_CONTAINS_NULLS` at validate() time) reads as "not specified" to anyone
  who doesn't already know that convention, indistinguishable from a key that was simply forgotten.

`schema_to_dict` fills all three gaps by augmenting `to_json()`'s own output: a `custom_checks`
list (name + docstring) for every check `to_json()` had to drop, `metadata` per column/dataframe
from `get_metadata()`, and an explicit `nullable: true`/`false` on every column, always.
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
        The dict `model.to_schema().to_json()` would produce, plus: a `metadata` key (from
        `model.get_metadata()`) on the dataframe and on each column that has one; a `custom_checks`
        key (list of `{name, description}`) on the dataframe and on each column that has a check
        `to_json()` couldn't describe declaratively; and an explicit `nullable` boolean on every
        column, even when `False` (which `to_json()` omits rather than states).
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
        column_dump['nullable'] = column_schema.nullable
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
