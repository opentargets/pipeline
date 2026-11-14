## croissant

`ot_croissant` builds the [Croissant](https://docs.mlcommons.org/croissant/docs/croissant-spec.html)
descriptor (`croissant.json`) for a release: one distribution per output dataset and
one record set describing its fields. It reads the parquet schemas of the output
datasets and fills in descriptions, keys and foreign keys from curated JSON in
`src/ot_croissant/assets/`.

## Where it runs in production

Not in the unified pipeline. `platform-output-support` (POS) installs it from this
monorepo by tag (`ot-croissant = { git = ..., subdirectory = "croissant", tag =
"croissant@v<version>" }` in its `pyproject.toml`) and runs it when publishing a
release. A change here reaches the published descriptor only after POS moves that pin
to a tag containing it, and moving the pin brings in every change since the old tag.
Check the pin rather than assuming it is recent.

## Curation assets

| File | Describes | Model (`models.py`) |
| --- | --- | --- |
| `assets/distribution.json` | each dataset: `nice_name`, `description`, `tags`, `key` | `DistributionAnnotation` |
| `assets/recordset/<dataset>.json` | each field of a dataset: `description`, `isPrimaryKey`, `foreign_key` | `RecordsetFieldAnnotation` |
| `assets/instance.json` | the Platform instances (public, ppp) | `InstanceAnnotation` |

- Every model is `extra='forbid'`: an unknown key fails validation.
- Field ids are relative to the dataset (`targetId`, not `evidence/targetId`). Struct
  and array-of-struct sub-fields need their own entries, keyed `parent/field`
  (e.g. `trialSponsor/agencyClass`).
- `foreign_key` is a full `dataset/field` id and must exist in the corpus.

## What ships when curation is missing

Nothing fails. Each gap is a WARNING in the log and a silent defect in the descriptor:

- **A field with no description** ships the literal `PLACEHOLDER for <field>
  description`. A description in the parquet field metadata takes precedence over the
  curation.
- **A curated key component missing from the schema** is dropped from the record set
  key. The key still validates, but may no longer be unique; check it against the
  data after an output schema change.
- **A dataset missing from `distribution.json`** ships "Description of the
  distribution '<id>' is not available."
- **A snake_case column** is warned about; output columns are camelCase.

So a pts change that adds, renames or removes an output column is a two-package
change: update `assets/recordset/<dataset>.json` in the same PR. A new dataset also
needs a `distribution.json` entry.

## Running and testing

```sh
cd croissant && uv run ot_croissant --help
cd croissant && uv run --frozen pytest -rxs
```

The tests check the curation files against the models, unique ids, the absence of
dataset prefixes, and that every foreign key points at an existing field. They do
**not** check curation against the real output schemas: to see the warnings for a
release, run `ot_croissant` on its output datasets and read the log.
