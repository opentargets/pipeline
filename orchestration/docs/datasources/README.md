# Input and data buckets

This document was updated on 2026-10-27.

## Input buckets

An input bucket is where we store downloaded files directly from datasource. Only datasources that can not be used directly in the pipelines,
are synced this way to Google Cloud Storage. The input buckets can not be publicly accessed outside OpenTargets organization.
The input bucket structure is not standardized.

- `gs://gwas_catalog_inputs` stores the GWAS Catalog summary statistics in raw format and files downloaded along the releases.

## Data buckets

A data bucket is location where we store the results that are referenced by the platform pipeline directly (ex. fine-mapping). These store the results from the processes that are cheap to run and can be stored in multiple copies.

```{bash}
gs://<data bucket>/
├── prod/
│   └── 20261021T143052/
│       ├── config/<dag>.yaml
│       ├── bin/
│       └── output/
├── test/
│   └── 20261021T150510/
├── fixtures/
└── cache/
```

The data bucket is divided into two environments (`prod` and `test`) along with `fixtures` and `cache`.

Each environment contains dated subfolder (ex. `20261021T143052` - UTC timestamp) that represents each run of the staging dag associated to the data bucket. The date is populated by the `{run_ts}` that is injected to the configuration from the `dag_run.run_after` during dag run. Individual runs do not override existing runs, each dag trigger creates new subfolder, but clearing steps in the dag does not. This means that data stored under dated subfolders can not be large (as we keep multiple copies).

A run folder contains:

- `config/`, the DAG config used by the run, with the paths of the run.
- `bin/`, any necessary scripts to run the dag (ex. install scripts for dataproc cluster)
- `output/`, the datasets written by the run.

Large data files (immutable) or incremental cache stores (like harmonised summary statistics for fine-mapping logs)  should be stored in the `input bucket` or `data bucket` cache, which is not dated, hence can change. The rule of thumb should be:

1) if the data needs to be downloaded once and it's size exceeds 100 Gb, store it under the isolated `input_bucket`.
2) if the data can be requested from the datasource directly and efficiently (gcs, aws, etc), do not download it, use the resource directly.
3) if the data needs to be downloaded once and it's size is below 100 Gb, store it under `cache` subdirectory.
4) if the update process is incremental (ex. GWAS Catalog susie fine-mapping), store cache under `cache` subdirectory.

## Test runs

A test run is a run of the DAG in the test environment. It runs the same steps as a prod run and stores its results in `test/`.

To keep the test runs isolated we use the same strategy as for prod environment where each test is isolated. Test fixtures should be stored under the `fixture` subfolder in the bucket. Test environment is not necessary for every dag. Tests should not depend on external buckets and shall be isolated to current bucket to ensure reproducibility. Fixtures should be slices of data and should only reflect the ability to perform integrated dag test.

A slice is a small copy of part of an input, made by hand, for example a few studies of a GWAS Catalog sync. Each slice is stored in its own folder, named after the time it was made. A new version of a slice goes into a new folder. Inside that folder, the slice keeps the path of the original input, so the steps read the same paths as in prod:

```{bash}
gs://gwas_catalog_top_hits/fixtures/20261026T153000/gwas_catalog_inputs/gentroutils/20260802/
                                    └─ when made ─┘ └─ the original input path ─────────────┘
```

Slices in `fixtures/` are read by test runs.

