# Finngen

This document was updated on 2026-10-29.

Bucket `gs://finngen_data` holds the FinnGen datasets ingested by Open Targets. Each DAG run writes to its own folder, `prod/<YYYYMMDDTHHMMSS>/`, with the DAG config of the run under `config/` and its outputs under `output/`. The docs of a run are under `docs/` of its run folder; the bucket has no top-level `docs/`.

```sh
gs://finngen_data/gwas/prod/<YYYYMMDDTHHMMSS>/       # FinnGen GWAS, finngen_ingestion DAG
gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/   # FinnGen multiome, finngen_multiome_ingestion DAG
```

## FinnGen GWAS

The SuSiE fine-mapping of [FinnGen Data Freeze 12](https://r12.finngen.fi/) is read from `gs://finngen-public-data-r12/` and ingested by the `finngen_ingestion` DAG into a study index and credible sets. The DAG config is `dags/config/finngen_ingestion.yaml`. The current release is `gs://finngen_data/gwas/prod/20260526T131014/`, see [FinnGen GWAS](finngen_gwas/README.md).

## FinnGen multiome

The single-cell cis-eQTL fine-mapping of the [FinnGen immune multiome atlas](https://www.nature.com/articles/s41586-026-11078-2) is read from `gs://finngen-public-data-multiome/eQTL/` and ingested by the `finngen_multiome_ingestion` DAG, with the gentropy step [`finngen_multiome_ingestion`](https://github.com/opentargets/gentropy/pull/1307). The DAG config is `dags/finngen_multiome_ingestion/config.yaml`. Each run writes to its own folder:

```sh
gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/config/   # DAG config of the run
gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/bin/      # gentropy CLI and cluster init script of the run
gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/output/study/
gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/output/credible_set/
gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/output/credible_set_variant/
```

## Changelog

### 2025-02-05

- [fix: updating info in finngen study index ingestion (#972)](https://github.com/opentargets/gentropy/pull/972)

### 2025-04-11

- chore: removal of r11 and r10 data to prevent data duplication.

### 2025-07-24

- [feat: updated finngen studyIndex to contain pubmedId upon ingestion](https://github.com/opentargets/issues/issues/3946)

### 2026-05-20

- [feat: rerun study index generation to include latest EFO changes](https://github.com/opentargets/issues/issues/4344)

### 2026-10-29

- feat: add the `finngen_multiome_ingestion` DAG, a StagingDagConfig DAG on gentropy `v4.0.0-dev.5` writing to `gs://finngen_data/multiome/prod/<YYYYMMDDTHHMMSS>/`.
- chore: move the FinnGen GWAS release (run of 2026-05-26) from `gs://finngen_data/r12/` to `gs://finngen_data/gwas/prod/20260526T131014/` (`config/`, `docs/`, `output/`), and split its docs into [FinnGen GWAS](finngen_gwas/README.md).
- chore: remove the top-level `gs://finngen_data/docs/`; the docs of each run are under `docs/` of its run folder.
- chore: remove `gs://finngen_data/100_3mb_carma/`, CARMA fine-mapping outputs that no DAG produces or reads.
