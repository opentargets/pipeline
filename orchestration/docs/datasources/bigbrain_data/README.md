# BigBrain

This document was updated on 2026-10-29.

[BigBrain](https://zenodo.org/records/17226890) is a brain-tissue molecular QTL meta-analysis. Its published SuSiE fine-mapping (PolyFun + SuSiE, L=5, in-sample LD from 3,884 European donors) is ingested as credible sets by the `bigbrain_ingestion` DAG, with the gentropy step [`bigbrain_finemapping_ingestion`](https://github.com/opentargets/gentropy/pull/1321). The DAG config is `dags/bigbrain_ingestion/config.yaml`.

Each run writes to its own folder:

```sh
gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/config/   # DAG config of the run
gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/bin/      # gentropy CLI and cluster init script of the run
gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/source/   # input files, copied from Zenodo
gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/output/{eqtl,sqtl}/study/
gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/output/{eqtl,sqtl}/credible_set/
gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/output/{eqtl,sqtl}/credible_set_variant/
```

The DAG has 3 steps:

1. `download`: copies the inputs from Zenodo into `source/`:
   - `BigBrain_cis_eQTL_finemapping.txt.gz` from [record 17226890](https://zenodo.org/records/17226890)
   - `BigBrain_cis_sQTL_finemapping.txt.gz` and `BigBrain_cis_sQTL_EUR_top_assoc.tsv.gz` (junction-to-gene mapping) from [record 17153730](https://zenodo.org/records/17153730)
2. `eqtl`: one study per gene.
3. `sqtl`: one study per intron junction. Junctions annotated to several genes are resolved with the target index of the platform run in `input.platform`.

## Changelog

### 2026-10-29

- feat: add the `bigbrain_ingestion` DAG, a StagingDagConfig DAG on gentropy `v4.0.0-dev.5`.
