# UK Biobank Pharma Proteomics Project (UKB-PPP)

This document was updated on 2026-10-31.

```sh
gs://ukb_ppp_eur_data/prod/20260526T131650/config/                                   # DAG configs of the run
gs://ukb_ppp_eur_data/prod/20260526T131650/docs/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/study_index/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/study_locus_lb_clumped/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/finemapping_manifests/20250129/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/finemapping_logs/20250129/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/credible_set_datasets/susie/20250129/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/credible_set_clean/20250129/
gs://ukb_ppp_eur_inputs/harmonised_summary_statistics/                               # input, not part of the run folder
```

Full description of the process can be found in https://github.com/opentargets/issues/issues/3234.

Data is produced by two DAGs, **ukb_ppp_eur_harmonisation** and **ukb_ppp_eur_finemapping**. The DAG configurations and topologies are in `config/ukb_ppp_eur_harmonisation.yaml` and `config/ukb_ppp_eur_finemapping.yaml` of the run folder (`gs://ukb_ppp_eur_data/prod/20260526T131650/config/`).

The run folder is named after the data freeze: the move of the release from the top level of `gs://ukb_ppp_eur_data/` into a dated folder.

Inputs of the run:

```sh
gs://gentropy-vault/ukb-ppp/
gs://gentropy-tmp/batch/output/ukb_ppp_eur/study_index.tsv       # deleted, rebuild with ./submit.py ukb_ppp_eur
gs://gentropy-tmp/batch/output/ukb_ppp_eur/summary_stats.parquet # deleted, rebuild with ./submit.py ukb_ppp_eur
gs://gnomad_data_2/gnomad_variant_index                          # deleted
gs://panukbb-ld-matrixes/UKBB.EUR.ldadj
gs://panukbb-ld-matrixes/ukb-diverse-pops-public-build-38/UKBB.EUR.ldadj.variant.b38
```

## Pre-steps

### 1. Mirror

- **Input.** Original data is hosted on Synapse.
- **Transformation.**
  - As we decided in the past, we want to keep the copy of the original data in the Vault.
  - Protocol is available here: https://github.com/opentargets/gentropy-vault/blob/main/datasets/ukb-ppp.md.
  - The protocol must be run manually.
- **Output.** The output of this step is kept forever in the Vault.

### 2. Preprocess

- **Input.** The mirrored data from the previous step.
- **Transformation.**
  - The data which we mirrored during the previous steps has to undergo several specific transformations which aren't achievable in Spark (especially the first one):
    - Extract gzipped per-chromosome files from inside the individual TAR files, decompress, partition by chromosome
    - Recreate the study ID. This is required because multiple rows in the study index can reuse the same summary stats file
    - Drop certain rows which don't have a corresponding summary stats file
  - This transformation is done using Google Batch. The code can be found in this new repository: [gentropy-input-support](https://github.com/opentargets/gentropy-input-support). The UKB PPP-specific part is this one: [data_sources.py](https://github.com/opentargets/gentropy-input-support/blob/dc5f8f7aee13a066933f3fd5b18a9b3a5ca71069/data_sources.py#L43-L103).
  - The command to run is `./submit.py ukb_ppp_eur` inside the `gentropy-input-support`
  - This step must also be triggered manually, how to do this is described in the repository.
- **Output.** Precursors of study index and summary stats datasets are output. Because we decided that we don't want to keep the data twice, the output of this step is only kept temporarily and is deleted after 60 days according to the _gs://gentropy-tmp_ bucket lifecycle rules.

## Orchestration steps

The harmonisation DAG has 2 steps, run one after another on one Dataproc cluster:

![ukb_ppp_eur_harmonisation](ukb_ppp_eur_harmonisation.svg)

1. ukb_ppp_eur_sumstat_preprocess
2. locus_breaker_clumping

The fine-mapping DAG has 3 steps, run one after another: the index and the fine-mapping jobs on Google Batch, and the quality control on one Dataproc cluster:

![ukb_ppp_eur_finemapping](ukb_ppp_eur_finemapping.svg)

1. generate_finemapping_index
2. finemapping_batch_job
3. ukb_ppp_eur_data_credible_set_qc

## ukb_ppp_eur_sumstat_preprocess

The step ([`ukb_ppp_eur_sumstat_preprocess.py`](https://github.com/opentargets/gentropy/blob/v2.3.0-rc.5/src/gentropy/ukb_ppp_eur_sumstat_preprocess.py)) builds the study index and harmonises the summary statistics.

The study index ([`study_index.py`](https://github.com/opentargets/gentropy/blob/v2.3.0-rc.5/src/gentropy/datasource/ukb_ppp_eur/study_index.py)) has one study per row of `study_index.tsv`. The output is `output/study_index/`.

The summary statistics ([`summary_stats.py`](https://github.com/opentargets/gentropy/blob/v2.3.0-rc.5/src/gentropy/datasource/ukb_ppp_eur/summary_stats.py), [`harmonise.py`](https://github.com/opentargets/gentropy/blob/v2.3.0-rc.5/src/gentropy/common/harmonise.py)) are harmonised one chromosome at a time:

- Chromosome 23 is renamed to X.
- Variants with an `INFO` below 0.8, or a minor allele frequency below 0.0001, are dropped.
- Each variant is matched to the gnomAD variant index (`gs://gnomad_data_2/gnomad_variant_index`) in its own allele order or in the flipped order. A flipped variant has its `BETA` negated and its allele frequency replaced by 1 − frequency.
- A flipped strand-ambiguous SNP (A/T or G/C) is dropped. Variants not in the variant index are dropped.
- Rows with a null position or a `BETA` of 0 are dropped.

## locus_breaker_clumping

The step ([`locus_breaker_clumping.py`](https://github.com/opentargets/gentropy/blob/v2.3.0-rc.5/src/gentropy/locus_breaker_clumping.py)) finds the loci of each study in the harmonised summary statistics:

- Locus-breaker clumping ([`locus_breaker_clumping.py`](https://github.com/opentargets/gentropy/blob/v2.3.0-rc.5/src/gentropy/method/locus_breaker_clumping.py)) joins the variants with a p-value below 1e-5 that are less than 250 kb apart into one locus, keeps the loci whose lead has a p-value below 1.7e-11, and extends each locus by 100 kb on both sides.
- A locus longer than 1.5 Mb is replaced by the windows of window-based clumping (500 kb, p-value below 1.7e-11) whose lead is inside it. Each such window spans 1.5 Mb centred on its lead.
- Loci overlapping the MHC region are dropped.
- The summary statistics of each locus are collected into `locus`.

The output is `output/study_locus_lb_clumped/`, partitioned by `studyLocusId`: 17,670 loci.

## generate_finemapping_index

The step writes a manifest of the loci in `output/study_locus_lb_clumped/` to `output/finemapping_manifests/20250129/`.

## finemapping_batch_job

The step fine-maps each locus of the manifest with [SuSiE-inf](https://github.com/opentargets/gentropy/blob/v2.0.2-rc.2/src/gentropy/susie_finemapper.py) in one Google Batch task per locus, and writes the credible sets to `output/credible_set_datasets/susie/20250129/` and the task logs to `output/finemapping_logs/20250129/`.

## ukb_ppp_eur_data_credible_set_qc

The step ([`credible_set_qc.py`](https://github.com/opentargets/gentropy/blob/v3.2.0/src/gentropy/credible_set_qc.py)) filters and deduplicates the credible sets, and writes them to `output/credible_set_clean/20250129/`, the dataset that the platform reads.

## Outputs

Schemas as read from the parquet files of the run. Nested columns show their `array<struct<...>>` types. "Empty" marks a column with no values in this run.

| Output | Rows | Files |
|---|---|---|
| `study_index` | 2,957 | 1 |
| `study_locus_lb_clumped` | 17,670 | 1 per locus |
| `credible_set_datasets/susie/20250129` | 1 folder per fine-mapped locus (16,974) | 1 or more per locus |
| `credible_set_clean/20250129` | 32,179 | 16 |

### study_index

One row per protein assay.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | `UKB_PPP_EUR_<gene>_<UniProt>_<OlinkId>_v<version>`, e.g. `UKB_PPP_EUR_A1BG_P04217_OID30771_v1`. |
| `studyType` | `string` | `pqtl`. |
| `projectId` | `string` | `UKB_PPP_EUR`. |
| `traitFromSource` | `string` | UKB-PPP protein identifier, `<gene>:<UniProt>:<OlinkId>:v<version>`. |
| `biosampleFromSourceId` | `string` | `UBERON_0001969` (blood plasma). |
| `geneId` | `string` | Ensembl gene of the protein. 2,937 distinct genes. |
| `hasSumstats` | `boolean` | `true`. |
| `summarystatsLocation` | `string` | TAR file of the study in the summary statistics mirror, `gs://gentropy-vault/ukb-ppp/…`. |
| `nSamples` | `integer` | Sample size of the study, 9,216 to 34,090. |
| `discoverySamples` | `array<struct<sampleSize: integer, ancestry: string>>` | One sample: `nSamples` and the ancestry `European`. |
| `ldPopulationStructure` | `array<struct<ldPopulation: string, relativeSampleSize: double>>` | `nfe` with a relative sample size of 1.0. |

### study_locus_lb_clumped

One row per locus, in one folder per `studyLocusId`.

| Column | Type | Description |
|---|---|---|
| `studyLocusId` | `string` | Identifier of the locus (the partition folder). |
| `studyId` | `string` | Study of the locus, as in `study_index`. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `chromosome` | `string` | Chromosome of the lead. |
| `position` | `integer` | Position of the lead on GRCh38. |
| `locusStart` | `integer` | Start of the locus. |
| `locusEnd` | `integer` | End of the locus. |
| `beta` | `double` | Effect size of the lead. |
| `standardError` | `double` | Standard error of the lead. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the lead (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the p-value of the lead. |
| `effectAlleleFrequencyFromSource` | `float` | Frequency of the effect allele of the lead. |
| `sampleSize` | `integer` | Sample size of the lead. |
| `qualityControls` | `array<string>` | Quality control flags of the locus. |
| `locus` | `array<struct<variantId: string, beta: double, pValueMantissa: float, pValueExponent: integer, standardError: double>>` | Summary statistics of every variant between `locusStart` and `locusEnd`. |

### credible_set_datasets/susie/20250129

One folder per fine-mapped locus, named after the `studyLocusId` of the locus in `study_locus_lb_clumped`. One row per credible set.

| Column | Type | Description |
|---|---|---|
| `studyLocusId` | `string` | Identifier of the credible set: a hash of `studyId`, `variantId` and `finemappingMethod`. |
| `studyId` | `string` | Study of the credible set, as in `study_index`. |
| `credibleSetIndex` | `integer` | Index of the credible set within the locus, 1 to 10. |
| `region` | `string` | Fine-mapped region, `<chromosome>:<start>-<end>`. |
| `variantId` | `string` | Lead variant, the variant of the credible set with the largest posterior probability. |
| `chromosome` | `string` | Chromosome of the lead. |
| `position` | `integer` | Position of the lead on GRCh38. |
| `finemappingMethod` | `string` | `SuSiE-inf`. |
| `credibleSetlog10BF` | `double` | Log10 Bayes factor of the credible set. |
| `purityMeanR2` | `double` | Mean r² between the variants of the credible set. |
| `purityMinR2` | `double` | Minimum r² between the variants of the credible set. |
| `zScore` | `double` | Z-score of the lead. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the lead. |
| `pValueExponent` | `integer` | Exponent of the p-value of the lead. |
| `locusStart` | `integer` | Start of the locus in `study_locus_lb_clumped`. |
| `locusEnd` | `integer` | End of the locus in `study_locus_lb_clumped`. |
| `beta` | `double` | Effect size of the lead, from `locus`. |
| `qualityControls` | `array<string>` | `Study locus finemapped without in-sample LD reference`. |
| `locus` | `array<struct<variantId: string, posteriorProbability: double, logBF: double, beta: double>>` | Variants of the credible set with their posterior probability, log Bayes factor and effect. |

### credible_set_clean/20250129

One row per credible set: the rows of `credible_set_datasets/susie/20250129` that pass the quality control, in the `StudyLocus` schema of gentropy `v3.2.0`.

| Column | Type | Description |
|---|---|---|
| `studyLocusId` | `string` | Identifier of the credible set, as in `credible_set_datasets/susie/20250129`. |
| `studyType` | `string` | Empty. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `chromosome` | `string` | Chromosome of the lead. |
| `position` | `integer` | Position of the lead on GRCh38. |
| `region` | `string` | Fine-mapped region, `<chromosome>:<start>-<end>`. 12,121 distinct regions. |
| `studyId` | `string` | Study of the credible set, as in `study_index`. Every study is in `study_index`. |
| `beta` | `double` | Effect of the lead, from `locus`. |
| `zScore` | `double` | Z-score of the lead. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the lead. |
| `pValueExponent` | `integer` | Exponent of the p-value of the lead. Every p-value is at most 1e-5. |
| `effectAlleleFrequencyFromSource` | `float` | Empty. |
| `standardError` | `double` | Empty. |
| `subStudyDescription` | `string` | Empty. |
| `qualityControls` | `array<string>` | `Study locus finemapped without in-sample LD reference`. |
| `finemappingMethod` | `string` | `SuSiE-inf`. |
| `credibleSetIndex` | `integer` | Index of the credible set within the locus, 1 to 10. |
| `credibleSetlog10BF` | `double` | Log10 Bayes factor of the credible set. |
| `purityMeanR2` | `double` | Mean r² between the variants of the credible set. |
| `purityMinR2` | `double` | Minimum r² between the variants of the credible set, at least 0.25. |
| `locusStart` | `integer` | Start of the locus. |
| `locusEnd` | `integer` | End of the locus. |
| `sampleSize` | `integer` | Empty. |
| `ldSet` | `array<struct<tagVariantId: string, r2Overall: double>>` | Empty. |
| `locus` | `array<struct<is95CredibleSet: boolean, is99CredibleSet: boolean, logBF: double, posteriorProbability: double, variantId: string, pValueMantissa: float, pValueExponent: integer, beta: double, standardError: double, r2Overall: double>>` | Variants of the credible set, 1 to 2,848 per credible set. Only `variantId`, `posteriorProbability`, `logBF` and `beta` have values. |
| `confidence` | `string` | Empty. |
| `isTransQtl` | `boolean` | Empty. |

## Changelog

### 2025-02-05

- [fix: repair SusieFinemapperStep to work with new SL schema #957](https://github.com/opentargets/gentropy/pull/957) resolved [issue](https://github.com/opentargets/issues/issues/3667)

### 2025-04-11

- chore: harmonised summary statistics were moved from `gs://ukb_ppp_eur_data/harmonised_summary_statistics` to `gs://ukb_ppp_eur_inputs/harmonised_summary_statistics`. Summary statistics are now in the archive bucket (cold storage).
- chore: removal of 2024-10-21 credible sets due to data duplication.

### 2026-09-16

- chore: moved the qc step from `credible_set_qc` dag to ukb_ppp_eur_finemapping dag.

### 2026-10-30

- chore: move the UKB-PPP release from the top level of `gs://ukb_ppp_eur_data/` to `gs://ukb_ppp_eur_data/prod/20260526T131650/` (`config/`, `docs/`, `output/`), following [opentargets/issues#4613](https://github.com/opentargets/issues/issues/4613). The DAGs write to `output/` of the run folder.
- chore: the credible set quality control paths of `ukb_ppp_eur_finemapping` follow `root_path` and `date`, so the `Test` environment no longer writes to the prod bucket.
- docs: rewrite the docs for the run folder; the fine-mapping LD reference is the Pan-UKBB `EUR` LD matrix.
