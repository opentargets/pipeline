"""Agreement between each dataset model and the pandera schema derived from it.

The rule checked: the derived pandera schema (before registered checks, which Pydantic cannot see)
accepts a one-row frame exactly when Pydantic, in strict mode, accepts that row. NaN is turned into
None for Pydantic first, since the derived schema treats NaN as null.

Rows are single-field mutations of the model's `example`, generated from its field tree for every
value at every depth: null, empty, NaN and infinities, each side of every bound, values outside a
`Literal`, pattern-breaking strings, lengths around every length limit, a value of the wrong kind,
an absent column and an undeclared one. Every discovered dataset is checked, plus a synthetic model
using every supported construct, so a construct no real dataset uses yet is covered too.
"""

import copy
import math
from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Annotated, Any, ClassVar, Literal

import pandera.errors as pe
import polars as pl
import pytest
from pydantic import Field, ValidationError

from pts.schemas.dataset import DatasetModel, PolarsDtype, PrimaryKey, StructModel, to_polars_schema
from pts.schemas.dataset.converter import to_base_pandera
from pts.schemas.dataset.discovery import discover_datasets
from pts.schemas.dataset.spec import Node, dataset_spec, walk


class _Leaf(StructModel):
    code: str = Field(pattern=r'^[A-Z]\d$')
    weight: float | None = Field(ge=0.0, lt=1.0)
    tags: list[Annotated[str, Field(min_length=2)]]


class _Branch(StructModel):
    name: str
    rank: Literal[1, 2, 3]
    leaves: list[_Leaf] = Field(max_length=2)


class _KitchenSink(DatasetModel):
    dataset_name: ClassVar[str] = '_kitchen_sink'
    dataset_description: ClassVar[str] = 'synthetic model using every supported construct'
    example: ClassVar[dict] = {
        'key': 'abcd',
        'count': 3,
        'ratio': 0.5,
        'flag': True,
        'kind': 'a',
        'loose': 'ab123',
        'branch': {'name': 'b', 'rank': 1, 'leaves': [{'code': 'A1', 'weight': 0.5, 'tags': ['xy']}]},
        'branches': [{'name': 'c', 'rank': 2, 'leaves': [{'code': 'B2', 'weight': None, 'tags': ['zz']}]}],
        'maybe_numbers': [1.5],
        'optional_note': 'note',
    }

    key: Annotated[str, PrimaryKey()] = Field(min_length=3, max_length=5)
    count: Annotated[int, PolarsDtype(pl.Int32)] = Field(gt=0, le=10)
    ratio: float | None = Field(gt=-1.0, lt=1.0)
    flag: bool
    kind: Literal['a', 'b']
    loose: str = Field(pattern=r'\d{3}')  # unanchored: matches anywhere, as Pydantic's does
    branch: _Branch | None
    branches: list[_Branch] = Field(min_length=1)
    maybe_numbers: list[float | None] | None
    optional_note: str | None = None


_MODELS = [_KitchenSink, *sorted(discover_datasets().values(), key=lambda m: m.dataset_name)]


@dataclass(frozen=True)
class Mutation:
    """A row to validate both ways.

    Attributes:
        label: what was changed.
        row: the row.
        inferred: columns whose dtype Polars infers from the value instead of taking the model's.
    """

    label: str
    row: dict[str, Any]
    inferred: frozenset[str] = field(default_factory=frozenset)


def mutations(model: type[DatasetModel]) -> Iterator[Mutation]:
    """Single-field mutations of `model.example`, for every value at every depth."""
    example = model.example
    for f in dataset_spec(model).fields:
        yield Mutation(f'{f.name}: absent', {k: v for k, v in example.items() if k != f.name})
        wrong = 1.5 if f.node.dtype in {pl.String(), pl.Boolean()} else 'text'
        yield Mutation(f'{f.name}: {wrong!r}', {**example, f.name: wrong}, frozenset({f.name}))
        for steps, path, node, value in locations(f.node, example[f.name], (f.name,), f.name):
            for label, candidate in candidates(node, value):
                yield Mutation(f'{path}: {label}', _replace(example, steps, candidate))
    yield Mutation('undeclared column', {**example, 'undeclared': 'x'}, frozenset({'undeclared'}))


def locations(node: Node, value: Any, steps: tuple, path: str) -> Iterator[tuple[tuple, str, Node, Any]]:
    """Every value inside `value` the example actually reaches: the first element of each list."""
    yield steps, path, node, value
    if node.element is not None and value:
        yield from locations(node.element, value[0], (*steps, 0), f'{path}[]')
    if node.kind == 'struct' and value is not None:
        for child in node.fields:
            yield from locations(child.node, value[child.name], (*steps, child.name), f'{path}/{child.name}')


def candidates(node: Node, value: Any) -> Iterator[tuple[str, Any]]:
    """Replacement values probing each rule on `node`, given its current example `value`."""
    yield 'null', None
    if node.element is not None:
        yield 'empty list', []
    if node.dtype.is_float():
        yield from [('NaN', math.nan), ('inf', math.inf), ('-inf', -math.inf)]
    c = node.constraints
    for bound in (c.ge, c.gt, c.le, c.lt):
        if bound is not None:
            for near in _around(bound, integer=node.dtype.is_integer()):
                yield repr(near), near
    if c.allowed is not None:
        yield 'not allowed', max(int(v) for v in c.allowed) + 1 if node.dtype.is_integer() else '__not_allowed__'
    if c.pattern is not None:
        yield from [('empty string', ''), ('prefixed', f'#{value}'), ('suffixed', f'{value}#')]
    for limit in (c.min_length, c.max_length):
        if limit is not None:
            for length in (limit - 1, limit, limit + 1):
                if length >= 0 and node.element is not None and value:
                    yield f'{length} elements', [value[0]] * length
                elif length >= 0 and node.element is None:
                    yield f'{length} characters', 'a' * length


def _around(bound: float, *, integer: bool) -> list[float]:
    if integer:
        return sorted({math.floor(bound) - 1, math.floor(bound), math.ceil(bound), math.ceil(bound) + 1})
    return [math.nextafter(bound, -math.inf), bound, math.nextafter(bound, math.inf)]


def _replace(row: dict[str, Any], steps: tuple, value: Any) -> dict[str, Any]:
    row = copy.deepcopy(row)
    target: Any = row
    for step in steps[:-1]:
        target = target[step]
    target[steps[-1]] = value
    return row


def _nan_to_none(value: Any) -> Any:
    if isinstance(value, float) and math.isnan(value):
        return None
    if isinstance(value, dict):
        return {k: _nan_to_none(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_nan_to_none(v) for v in value]
    return value


def pydantic_accepts(model: type[DatasetModel], row: dict[str, Any]) -> bool:
    try:
        model.model_validate(_nan_to_none(row))
    except ValidationError:
        return False
    return True


def pandera_accepts(model: type[DatasetModel], mutation: Mutation) -> bool:
    schema = to_polars_schema(model)
    frame = pl.DataFrame([
        pl.Series(name, [value], dtype=schema[name], strict=True)
        if name in schema and name not in mutation.inferred
        else pl.Series(name, [value])
        for name, value in mutation.row.items()
    ])
    try:
        to_base_pandera(model).validate(frame, lazy=True)
    except pe.SchemaErrors:
        return False
    return True


@pytest.mark.parametrize('model', _MODELS, ids=lambda m: m.dataset_name)
def test_pandera_agrees_with_pydantic_on_every_mutation(model: type[DatasetModel]) -> None:
    checked = list(mutations(model))
    disagreements = [
        f'{m.label}: pydantic {"accepts" if p else "rejects"}, pandera {"accepts" if q else "rejects"}'
        for m in checked
        if (p := pydantic_accepts(model, m.row)) != (q := pandera_accepts(model, m))
    ]

    assert pydantic_accepts(model, model.example)
    assert pandera_accepts(model, Mutation('example', model.example))
    assert not disagreements, '\n'.join(disagreements)
    assert len(checked) > 3 * len(dataset_spec(model).fields)


@pytest.mark.parametrize('model', _MODELS, ids=lambda m: m.dataset_name)
def test_example_reaches_every_value(model: type[DatasetModel]) -> None:
    """Without this, an empty list in an example would silently skip every rule inside it."""
    spec = dataset_spec(model)
    reached = {path for f in spec.fields for _, path, _, _ in locations(f.node, model.example[f.name], (), f.name)}

    assert [path for path, _ in walk(spec) if path not in reached] == []
