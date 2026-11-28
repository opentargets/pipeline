## datamodels

`ot_datamodels` (distribution `ot-datamodels`) declares each output dataset of the
pipeline once, as a Pydantic model. Everything else is derived from the model: the
Polars schema (`to_polars_schema`), the documentation JSON
(`python -m ot_datamodels.docs <dir>`) and, in pts, the pandera schema validated
against. The models are a specification: Pydantic never validates data with them.

It depends on `pydantic` and `polars` only. It imports neither pts nor pandera, and
`tests/test_isolation.py` keeps it so.

## Layout

| Module | Holds |
| --- | --- |
| `base` | `DatasetModel`, `StructModel` and the `Annotated` markers |
| `spec` | the field tree every reader uses, and the supported subset of Pydantic |
| `polars` | `to_polars_schema` |
| `docs` | documentation JSON |
| `discovery` | `discover_datasets()`: every concrete model under `ot_datamodels.outputs`; `discover_models()`: those and the views |
| `derive` | `from_output` and `row_of`: view columns copied from output columns |
| `serving` | `Serving` and `ClickHouseDictionary`: how a view is served in ClickHouse |
| `vocabularies` | the closed vocabularies stored as Enums, one `Literal` per vocabulary, with its source |
| `outputs/` | one module per dataset (or family); `patterns.py` holds shared regexes |
| `views/` | one module per pts view (`view/<dataset_name>`), named after it; not release outputs, so not in `discover_datasets()` |

## What a model may state

Only a strict subset of Pydantic is supported (`spec`); anything outside it raises
`SchemaDefinitionError` rather than going unenforced.

- **Types:** `str`, `bool`, `int`, `float`, `datetime.date`, `Literal`, lists, structs
  (`StructModel`) and `| None`.
- **Constraints:** `ge`/`gt`/`le`/`lt`, `pattern`, `min_length`/`max_length`, on a
  column, a struct field or a list element alike.
- **Markers:** `PrimaryKey` (validated as unique), `Reference` (a foreign key,
  recorded but not enforced), `Bioregistry` and `Note` (documentation only),
  `PolarsDtype` (a narrower dtype such as `Int32`, or `pl.Enum` for a stored Enum
  whose categories are the `Literal`'s values in order, or `pl.Enum([...])` for the Enum of a
  whole vocabulary holding the `Literal`'s values in the same order). A ClickHouse type is not a
  marker: a served view declares its types in `Serving.column_types` (Views, below), which pos's DDL
  generation reads and pts does not; `LowCardinality` on a nullable value is emitted as
  `LowCardinality(Nullable(..))`.
- **Row checks:** rules across columns of one row go in `row_checks`, as `RowCheck`s.
  Rules across datasets cannot be in a model, since a model sees only its own dataset.
- **Closed vocabularies:** a `Literal` only where the list holds by construction (an
  enum in the code, constants the code writes, a fixed standard vocabulary).
  Input data passed through stays open: a non-empty string, or a pattern, unless David decides to
  close it (`TranscriptBiotype`, `ClinicalSource`): a new value then fails validation on purpose. A vocabulary
  consumers filter, join or order on is an Enum: its `Literal` lives in `vocabularies`
  (complete, ordered, its source cited) and a field uses it as
  `Annotated[Vocabulary, PolarsDtype(pl.Enum)]`, or `Annotated[Literal['eva'], enum_of(Datasource)]`
  for part of one. A new value is a model change; ClickHouse numbers the Enum in that order, so append
  where the order does not matter. `Datasource` and `Datatype` hold the evidence (and association)
  values only; another dataset's sources get a vocabulary of their own (`ExpressionDatasource`, ...).

A model states what holds for the data's meaning, not for one release's values: the
next release must not fail because a new value appeared. Numbers are bounded at
natural limits (`[0, 1]`, `>= 0`), never at the observed min/max.

`products` (default `('platform', 'ppp')`) names the products a dataset is loaded for; set
`('ppp',)` on a PPP-only dataset so pos keeps it out of the Platform schema. An integer
ClickHouse type (a `column_types` entry) narrower than the dtype, or unsigned,
needs `ge`/`le` inside its range, so a value that would wrap in ClickHouse fails pts validation.

## Adding a dataset

No generator exists; write the model by hand.

1. Add `outputs/<dataset>.py` with one `DatasetModel` subclass: `dataset_name` (also
   where it is written, `output/<name>`), `dataset_title`, `dataset_description` and
   `example` (one valid row).
2. Declare each column as a field with a `description`; struct columns are
   `StructModel` subclasses. Reuse `outputs/patterns.py` and `outputs/common.py`.
3. `discover_datasets()` finds it; nothing is registered.
4. Run the tests here, then pts's and pos's.

## Views

A view's `dataset_name` ends in `_view`, its module is named after it and its class ends in `View`
(`TargetView` in `views/target_view.py`); pts writes it to `view/<dataset_name>`. pts step names are
not renamed with it (they are DAG task ids). The one exception is `views/evidence_views.py`, which generates
the per-source evidence views from the `evidence_*` outputs (`evidence_<source>_view`, a whole copy, and its
index views `_by_disease_and_target_view` and, for an output with `variantId`, `_by_variant_view`).

Every view declares how pos serves it, on the class:

```python
class TargetView(Target):
    dataset_name: ClassVar[str] = 'target_view'
    clickhouse: ClassVar[Serving] = Serving(
        engine='rocksdb',
        dictionaries=(ClickHouseDictionary('target_symbols', key='id', attributes=('approvedSymbol',)),),
    )
```

`Serving` holds only the engine (`rocksdb` or `mergetree`), a MergeTree's `order_by` (required there,
refused on RocksDB), the dictionaries built from the table and `column_types`: the ClickHouse type of a
primitive's non-null value where it differs from its dtype's, by `spec.walk` path (`column`,
`column[]` for a list element, `column/field` for a struct field: `constraint[]/obs`). The served table is the whole view,
named `dataset_name`; pos derives the rest from the model: the RocksDB key is its one `PrimaryKey`, the
landing table `<dataset_name>_log`, the input `view/<dataset_name>` and the products the model's
(`pos/AGENTS.md`, ClickHouse landing tables). A column the API does not read is dropped from the view,
not from the table. Renaming a view renames its ClickHouse table: add the rename to
`pos/docs/api_follow_ups.csv`. `tests/test_serving.py` checks the declarations against the columns
(each `column_types` path names a primitive value of the view); output models have no `Serving`. A view
lists every ClickHouse type it serves in `column_types`, those of the columns it copies and nested
paths included.

## Deriving a view's columns

A view column that copies an output column is derived from the output model, never
retyped. A whole output is copied by subclassing it (`class TargetView(Target)`); a
column by `from_output`, a nested row of chosen columns by `row_of`. Both return a type:
bind it to a module-level name and annotate with it (`ty` rejects a call inside an
annotation):

```python
Score = from_output(Evidence, 'score')
HomologueRow = row_of(Homology, ['targetId', 'type'], name='HomologueRow')

class AssociationView(DatasetModel):
    rowScore: Score
    homologues: list[HomologueRow]
```

The copy keeps the type, markers, constraints and description; the column may be
renamed. Nullability is the one override (`nullable=True`, `row_of(..., nullable={...})`).
`PrimaryKey` is dropped unless `from_output(..., primary_key=True)`; `row_of` always drops
it. Computed columns are declared in the view model. A list element cannot carry a description, so a
list of copied values (`transcriptIds`) is declared and `tests/test_views.py` checks it against the
output; that test lists every derived value of every view.

## Validation

Validation lives in pts: a model's pandera schema (`to_pandera`) and
`validate_dataset` (`pts.schemas.dataset.validate`) run there, and each pts transformer
validates what it writes against its model.

A change here also needs the pts and pos tests: pts checks every output task
against its model and the models against its enums, and pos builds the release's croissant
descriptor from the models (a field with no description fails pos's tests).

Run `uv run --frozen pytest`, `ruff check` and `ty check` from this directory.
