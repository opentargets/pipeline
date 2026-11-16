# UK Biobank Pharma Proteomics Project (UKB-PPP)

This document was updated on 2026-11-01.

```sh
gs://ukb_ppp_eur_data/prod/20261026T161010/config/       # migration scripts of the run
gs://ukb_ppp_eur_data/prod/20261026T161010/docs/
gs://ukb_ppp_eur_data/prod/20261026T161010/output/study/
gs://ukb_ppp_eur_data/prod/20261026T161010/output/credible_set/
gs://ukb_ppp_eur_data/prod/20261026T161010/output/credible_set_variant/
```

Full description of the process can be found in https://github.com/opentargets/issues/issues/3234.

Data in this run folder is the UKB-PPP release `gs://ukb_ppp_eur_data/prod/20260526T131650/`, migrated by hand to the `CredibleSet` and `CredibleSetVariant` datasets of gentropy v4 ([opentargets/gentropy#1326](https://github.com/opentargets/gentropy/pull/1326)). The migration converts the released credible sets without fine-mapping them again. It uses the one-off scripts of [opentargets/gentropy#1340](https://github.com/opentargets/gentropy/pull/1340), at commit [`7bb17a92`](https://github.com/opentargets/gentropy/tree/7bb17a9237fa77bd66691aa722c01b34f8be806a/utils), which are copied to `config/` of the run folder. The DAGs that produced the release, `ukb_ppp_eur_harmonisation` and `ukb_ppp_eur_finemapping`, are described in `gs://ukb_ppp_eur_data/prod/20260526T131650/docs/README.md`.

Inputs of the run:

```sh
gs://ukb_ppp_eur_data/prod/20260526T131650/output/credible_set_clean/20250129/
gs://ukb_ppp_eur_data/prod/20260526T131650/output/study_index/
gs://ukb_ppp_eur_inputs/harmonised_summary_statistics/
```

The migration read the release before its datasets were moved under `output/`, from `gs://ukb_ppp_eur_data/prod/20260526T131650/credible_set_clean/20250129/` and `…/study_index/`.

## Steps

1. **Migration.** [`susie_credible_set_migration.sh`](https://github.com/opentargets/gentropy/blob/7bb17a9237fa77bd66691aa722c01b34f8be806a/utils/susie_credible_set_migration.sh) `run ukb_ppp_eur` runs [`susie_credible_set_migration.py`](https://github.com/opentargets/gentropy/blob/7bb17a9237fa77bd66691aa722c01b34f8be806a/utils/susie_credible_set_migration.py) on a Dataproc cluster, which writes `credible_set` and `credible_set_variant` (50 partitions), and copies the released study index to `study`. The job ends with checks of its output and fails when one does not hold.
2. **Single-file rewrite.** [`susie_credible_set_coalesce.py`](https://github.com/opentargets/gentropy/blob/7bb17a9237fa77bd66691aa722c01b34f8be806a/utils/susie_credible_set_coalesce.py) reads the three datasets through the gentropy dataset classes, writes each to one parquet file in this run folder, and checks that they hold the same rows as the migration output.

The column-by-column check of the output, [`susie_credible_set_migration_check.py`](https://github.com/opentargets/gentropy/blob/7bb17a9237fa77bd66691aa722c01b34f8be806a/utils/susie_credible_set_migration_check.py), was run on test data (`gs://ot_orchestration/tests/susie_credible_set_migration`).

The multi-file output of the migration, `gs://ukb_ppp_eur_data/prod/20261026T154424/`, was deleted after the rewrite.

The migration maps each credible set and its members as follows:

| Input | Output |
|---|---|
| `locus[].posteriorProbability` (SuSiE-inf alpha) | `credible_set_variant.posteriorProbability` |
| `locus[].logBF` | `credible_set_variant.logBayesFactor` |
| `locus[].beta` (SuSiE-inf mu) | `credible_set_variant.conditionalBeta` |
| `credibleSetlog10BF` · ln 10 | `credible_set.logBayesFactor` |
| harmonised summary statistics row of the member | `credible_set_variant.pValueMantissa`, `pValueExponent`, `reportedBeta`, `reportedStandardError`, `sampleSize` |
| study index | `credible_set.studyType` |
| member with the highest posterior probability, ties broken by `variantId` | `credible_set.leadVariantId` |

## Outputs

Schemas as read from the parquet files of the run. "Empty" marks a column with no values in this run.

| Output | Rows | Files |
|---|---|---|
| `study` | 2,957 | 1 |
| `credible_set` | 32,179 | 1 |
| `credible_set_variant` | 408,970 | 1 |

`credible_set` and `credible_set_variant` hold the same credible sets and members as `credible_set_clean/20250129` of the release. Every credible set has a study in `study`, and every variant has a credible set in `credible_set`.

One credible set has a different lead from the release: in `UKB_PPP_EUR_CREG1_O75629_OID21515_v1`, region `12:101068376-102568376`, the members `12_101772435_G_A` (the lead of the release) and `12_101772432_AG_A` have the same posterior probability, 0.13011703321030177, and the tie is broken by `variantId`, which picks `12_101772432_AG_A`.

### study

One row per protein assay: the rows of the released `study_index`, in the `StudyIndex` schema of gentropy v4. The 11 columns below hold the same values as the released study index.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | `UKB_PPP_EUR_<gene>_<UniProt>_<OlinkId>_v<version>`, e.g. `UKB_PPP_EUR_A1BG_P04217_OID30771_v1`. |
| `studyType` | `string` | `pqtl`. |
| `projectId` | `string` | `UKB_PPP_EUR`. |
| `traitFromSource` | `string` | UKB-PPP protein identifier, `<gene>:<UniProt>:<OlinkId>:v<version>`. |
| `biosampleFromSourceId` | `string` | `UBERON_0001969` (blood plasma). |
| `geneId` | `string` | Ensembl gene of the protein. |
| `hasSumstats` | `boolean` | `true`. |
| `summarystatsLocation` | `string` | TAR file of the study in the summary statistics mirror, `gs://gentropy-vault/ukb-ppp/…`. |
| `nSamples` | `integer` | Sample size of the study, 9,216 to 34,090. |
| `discoverySamples` | `array<struct<sampleSize: integer, ancestry: string>>` | One sample: `nSamples` and the ancestry `European`. |
| `ldPopulationStructure` | `array<struct<ldPopulation: string, relativeSampleSize: double>>` | `nfe` with a relative sample size of 1.0. |

The other 19 columns are empty: `traitFromSourceMappedIds`, `diseaseIds`, `biosampleId`, `pubmedId`, `publicationTitle`, `publicationFirstAuthor`, `publicationDate`, `publicationJournal`, `backgroundTraitFromSourceMappedIds`, `backgroundDiseaseIds`, `initialSampleSize`, `nCases`, `nControls`, `cohorts`, `replicationSamples`, `qualityControls`, `analysisFlags`, `condition` and `sumstatQCValues`.

### credible_set

One row per credible set.

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Identifier of the credible set: the md5 hash of `studyId`, `chromosome`, `locusStart`, `locusEnd`, `leadVariantId` and `fineMappingMethod`. |
| `chromosome` | `string` | Chromosome of the locus. |
| `locusStart` | `integer` | Start of the fine-mapped locus. |
| `locusEnd` | `integer` | End of the fine-mapped locus. |
| `studyId` | `string` | Study of the credible set, as in `study`. 2,378 studies. |
| `studyType` | `string` | `pqtl`, from `study`. |
| `leadVariantId` | `string` | Lead variant, the member with the highest posterior probability, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `fineMappingMethod` | `string` | `SuSiE-inf`. |
| `logBayesFactor` | `double` | Natural log Bayes factor of the credible set, `credibleSetlog10BF` · ln 10. |
| `confidence` | `tinyint` | Empty. |
| `purityMeanR2` | `double` | Mean r² between the members of the credible set. |
| `purityMinR2` | `double` | Minimum r² between the members of the credible set, at least 0.25. |
| `qualityControls` | `array<string>` | `Study locus finemapped without in-sample LD reference`. |

### credible_set_variant

One row per member of a credible set.

| Column | Type | Description |
|---|---|---|
| `credibleSetId` | `string` | Credible set of the variant, as `id` in `credible_set`. |
| `variantId` | `string` | Variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the variant in the harmonised summary statistics (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the p-value of the variant. |
| `effectAlleleFrequency` | `float` | Empty: the harmonised summary statistics hold no allele frequency. |
| `sampleSize` | `integer` | Sample size of the variant in the harmonised summary statistics. |
| `reportedBeta` | `double` | Effect size of the variant in the harmonised summary statistics, for the alternate allele. |
| `reportedStandardError` | `double` | Standard error of `reportedBeta`. |
| `posteriorProbability` | `float` | SuSiE-inf posterior probability of the variant (alpha). |
| `logBayesFactor` | `float` | Natural log Bayes factor of the variant. |
| `r2Overall` | `float` | Empty: not stored by the fine-mapping. |
| `conditionalBeta` | `float` | SuSiE-inf posterior mean effect of the variant (mu). |
| `conditionalStandardError` | `float` | Empty: not stored by the fine-mapping. |
| `rescaledAbsBeta` | `float` | Empty. |
| `rescaledStandardError` | `float` | Empty. |
| `rescaledDirection` | `tinyint` | Empty. |
| `qualityControls` | `array<string>` | Empty for every variant: every member has summary statistics. |

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

### 2026-11-01

- chore: migrate the UKB-PPP release `prod/20260526T131650/` by hand to `CredibleSet` and `CredibleSetVariant` with the scripts of [opentargets/gentropy#1340](https://github.com/opentargets/gentropy/pull/1340), into `gs://ukb_ppp_eur_data/prod/20261026T161010/` (`config/`, `docs/`, `output/`).
- chore: remove `gs://ukb_ppp_eur_data/prod/20261026T154424/`, the multi-file output of the migration.
- docs: rewrite the docs for the migrated run; the docs of the release stay in `gs://ukb_ppp_eur_data/prod/20260526T131650/docs/`.
