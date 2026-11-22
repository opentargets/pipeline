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
| `discovery` | `discover_datasets()`: every concrete model under `ot_datamodels.outputs` |
| `outputs/` | one module per dataset (or family); `patterns.py` holds shared regexes |

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
  whose categories are the `Literal`'s values in order), `ClickHouseType` (the
  ClickHouse type of the non-null value, where it differs from the dtype; read by pos's
  DDL generation from phase 5 piece 2, ignored by pts; `LowCardinality` on a nullable
  field is emitted as `LowCardinality(Nullable(..))`).
- **Row checks:** rules across columns of one row go in `row_checks`, as `RowCheck`s.
  Rules across datasets cannot be in a model, since a model sees only its own dataset.
- **Closed vocabularies:** a `Literal` only where the list holds by construction (an
  enum in the code, constants the code writes, a small stable standard vocabulary).
  Input data passed through stays open: a non-empty string, or a pattern.

A model states what holds for the data's meaning, not for one release's values: the
next release must not fail because a new value appeared. Numbers are bounded at
natural limits (`[0, 1]`, `>= 0`), never at the observed min/max.

`products` (default `('platform', 'ppp')`) names the products a dataset is loaded for; set
`('ppp',)` on a PPP-only dataset so pos keeps it out of the Platform schema. An integer
`ClickHouseType` narrower than the dtype, or unsigned, needs `ge`/`le` inside its range, so a value
that would wrap in ClickHouse fails pts validation.

## Adding a dataset

No generator exists; write the model by hand.

1. Add `outputs/<dataset>.py` with one `DatasetModel` subclass: `dataset_name` (also
   where it is written, `output/<name>`), `dataset_title`, `dataset_description` and
   `example` (one valid row).
2. Declare each column as a field with a `description`; struct columns are
   `StructModel` subclasses. Reuse `outputs/patterns.py` and `outputs/common.py`.
3. `discover_datasets()` finds it; nothing is registered.
4. Run the tests here, then pts's and pos's.

## Validation

Validation lives in pts: a model's pandera schema (`to_pandera`) and
`validate_dataset` (`pts.schemas.dataset.validate`) run there, and each pts transformer
validates what it writes against its model.

A change here also needs the pts and pos tests: pts checks every output task
against its model and the models against its enums, and pos needs a croissant record set
entry for every new column.

Run `uv run --frozen pytest`, `ruff check` and `ty check` from this directory.
