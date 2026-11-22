## pts

`pts` has **two dynamic extension mechanisms**, both resolved by name at runtime
from `config.yaml`:

- **Transformers** (`src/pts/transformers/<name>.py`) — the `Transform` task
  imports `pts.transformers.<name>` and calls a function of the same name
  `(source, destination, settings, config)`. Local polars transforms. A dotted name
  (`evidence.encore`) reaches a subpackage, calling its last part; an optional
  `<name>_check` function in the module runs in the validation phase.
- **Pyspark jobs** (`src/pts/pyspark/<name>.py`) — the `Pyspark` task imports
  `pts.pyspark.<name>` and calls a function of the same name. Spark jobs.

To add one: create the module with a same-named callable, then wire it into
`config.yaml`.

Prefer a polars transformer for new work, unless the scale of the task requires a
cluster. Stay eager in small, self-contained steps; go lazy for big chains and
anything shared across datasources. When porting a pyspark job to polars, use the
`port-pyspark-to-polars` skill.

## Testing and type checking

- Run tests through `uv run`: calling `.venv/bin/python` directly leaves Spark
  workers on whichever `python` is on `PATH`, and every Spark test fails with
  `PYTHON_VERSION_MISMATCH`.
- Spark column functions give `ty` false positives. `pyproject.toml` already ignores
  `missing-argument`/`invalid-argument-type` in `src/pts/pyspark/**` and tests, so
  those files need no inline ignores for them.

## How a step runs in production

Read `orchestration/AGENTS.md` before porting a step off Spark or naming a new
transformer in `config.yaml`: it decides which steps run on Dataproc, and which config
changes need a `pts_version` bump.

## Rules

- **Write outputs through otter storage** (`StorageHandle`, or a library that
  speaks `gs://`, as polars does). In production every relative destination is a
  `gs://` URI; `pathlib` collapses it to a local `gs:/bucket/...` path, the write
  succeeds, the step goes green, and the release is missing the file.
- **An output column added, renamed or removed changes croissant curation in the
  same PR.** Nothing fails without it; `pos/AGENTS.md` says what to edit under `pos/src/pos/croissant/assets/`.
- Output datasets are declared once, as models in `datamodels/` (`ot_datamodels.outputs`);
  a transformer's `validate` imports its model from there (`datamodels/AGENTS.md`).
- A polars transformer writes a modelled output through `conform(frame, Model)`
  (`pts.schemas.dataset.conform`), which stores its closed vocabularies as Enums. A step
  reading an output joins, concatenates or applies `str` functions to those columns only
  after `without_enums`. `scan_parquet(schema=...)` with `pl.String` for an Enum column
  fails on such a file: read it with its own dtypes, then `without_enums`. A partial
  rerun that mixes intermediates written before and after a column became an Enum
  fails on concat (different categories); rerun the whole chain.
- Input and intermediate schemas (`pl.Schema`, schema dicts, reused `pl.Struct` dtypes)
  go in `src/pts/schemas/`, one module per step (follow `drug_molecule.py`), not in
  the transformer.
- Read spreadsheets with `pts.transformers.utils.dataset.read_spreadsheet`.
