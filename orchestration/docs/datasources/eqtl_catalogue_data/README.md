# EQTL Catalogue

This document was updated on 2026-11-17.

```sh
gs://eqtl_catalogue_data/r8/prod/20261117T103813/bin/                                 # gentropy cli.py and cluster init script of the run
gs://eqtl_catalogue_data/r8/prod/20261117T103813/config/eqtl_catalogue_ingestion.yaml # DAG config of the run
gs://eqtl_catalogue_data/r8/prod/20261117T103813/docs/
gs://eqtl_catalogue_data/r8/prod/20261117T103813/output/study_index/
gs://eqtl_catalogue_data/r8/prod/20261117T103813/output/credible_set/
gs://eqtl_catalogue_data/r8/prod/20261117T103813/output/credible_set_variant/
```

`r8/prod/20261117T103813/` is the DAG output, unpatched. It replaces the v4 run `r8/prod/20261105T130431/`, described in the README of earlier commits, whose credible set variants had no `effectAlleleFrequency`, so PTS could not rescale their effects ([opentargets/issues#4683](https://github.com/opentargets/issues/issues/4683)). Apart from that frequency, the two runs hold the same data. The run `r8/prod/20261117T083511/`, on gentropy `v4.0.0-dev.14`, is superseded: 76 of its frequencies were 1 or above (see below).

The run is in the gentropy v4 schema: credible sets (`credible_set`) and their variants (`credible_set_variant`). PIS stages it for PTS (`staging_eqtl_catalogue_ts` in `pis/config.yaml`).

Data contained in the run is the [eQTL Catalogue](https://www.ebi.ac.uk/eqtl/) release 8 SuSiE fine-mapping, ingested as a study index, credible sets and credible set variants. The eQTL Catalogue r8 already contains GTEx v10, IBDverse and MAGE, so they are not ingested on their own.

Data is produced by the **eqtl_catalogue_ingestion DAG**. The DAG configuration is in `config/eqtl_catalogue_ingestion.yaml` of the run folder. The run uses gentropy [`v4.0.0-dev.15`](https://github.com/opentargets/gentropy/tree/v4.0.0-dev.15) (`GENTROPY_REF`), the gentropy dev branch after `v4.0.0-rc.3` with the effect allele frequency of the credible set variants ([opentargets/gentropy#1349](https://github.com/opentargets/gentropy/pull/1349)), null where the allele count reaches 2 × `sample_size` ([opentargets/gentropy#1353](https://github.com/opentargets/gentropy/pull/1353)), and the [`eqtl_catalogue`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.15/src/gentropy/eqtl_catalogue.py) step. It also has the `pubmedId` fix ([opentargets/gentropy#1334](https://github.com/opentargets/gentropy/pull/1334)), one full SuSiE row per variant, region and trait ([#1338](https://github.com/opentargets/gentropy/pull/1338)) and a negative region start parsed as a locus start of 1 ([#1344](https://github.com/opentargets/gentropy/pull/1344)). The links below refer to that tag.

Inputs of the run:

```sh
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/eQTL_Catalogue_r8_260626/susie/QTS*/QTD*/QTD*.credible_sets.parquet
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/eQTL_Catalogue_r8_260626/susie/QTS*/QTD*/QTD*.lbf_variable.parquet
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/eQTL_Catalogue_r8_260626/susie_batches/QTS*/QTD*/QTD*.full_susie.parquet
https://raw.githubusercontent.com/eQTL-Catalogue/eQTL-Catalogue-resources/fd99d45d3e433dcdf3c3d0acb4db0b7d5e8a7bd3/data_tables/dataset_metadata_r8.tsv
```

The SuSiE results are the [qtlmap](https://github.com/eQTL-Catalogue/qtlmap) outputs in Parquet format, one folder per study (`QTS*`) and dataset (`QTD*`): 1,205 datasets with credible sets (6.6 GB) and log Bayes factors (1.5 TB), and 1,193 with the full SuSiE posterior (5.5 TB). The dataset metadata is the eQTL Catalogue r8 table of datasets, pinned to a commit.

The DAG has 1 step, run on one Dataproc cluster:

![eqtl_catalogue_ingestion](eqtl_catalogue_ingestion.svg)

1. ingestion

## ingestion

The step, gentropy `eqtl_catalogue` ([`eqtl_catalogue.py`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.15/src/gentropy/eqtl_catalogue.py)) joins the SuSiE credible sets, their log Bayes factors, the full SuSiE posterior and the dataset metadata ([`finemapping.py`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.15/src/gentropy/datasource/eqtl_catalogue/finemapping.py), [`study_index.py`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.15/src/gentropy/datasource/eqtl_catalogue/study_index.py)):

- One study per dataset and molecular trait. `studyType` comes from the quantification method (gene expression, exon, transcript and microarray give `eqtl`; Leafcutter and MAJIQ give `sqtl`; txrevise gives `tuqtl`; aptamer gives `pqtl`), with an `sc` prefix for single-cell datasets.
- One credible set per SuSiE component (1 to 10) of a molecular trait in a region. The lead is the member with the top posterior probability, ties broken by variant id.
- The region is the gene TSS ± 1 Mb, `chrN:start-end`. For genes in the first megabase of a chromosome its start is negative ([eQTL-Catalogue/qtlmap#29](https://github.com/eQTL-Catalogue/qtlmap/issues/29)); `locusStart` is then 1.
- The full SuSiE posterior of QTS000038 (OneK1K) lists some members twice, with moments that differ only by floating point noise. One row per variant, region and trait is kept.
- Variants with alleles other than A, C, G and T (e.g. `<DEL>`, `<DUP>` in AFR_LCL/QTS000044 and MAGE/QTS000055) are removed from the credible sets, as they gave duplicate entries.
- Some studies (e.g. CommonMind/QTS000008, Kim-Hellmuth/QTS000042) report the SuSiE PIP instead of the per-component alpha. Their posterior probabilities are recomputed from the log Bayes factors, so they sum to at most 1 per credible set ([opentargets/issues#4496](https://github.com/opentargets/issues/issues/4496)).
- The conditional statistics of a member are the SuSiE posterior mean and standard deviation of the credible set's own effect, from the full SuSiE posterior, on the scale SuSiE was fitted on.
- The effect allele is the ALT allele, and `ac` in the credible sets is its count. The credible sets have no total allele count `an`, which the eQTL Catalogue gives as 2 × `sample_size` ([eQTL-Catalogue/eQTL-Catalogue-resources#63](https://github.com/eQTL-Catalogue/eQTL-Catalogue-resources/issues/63#issuecomment-6850159973)). So `effectAlleleFrequency` is `ac / (2 × sample_size)` on chromosomes 1 to 22, and null on chrX, as datasets count the chrX alleles differently (Alasoo_2018 counts each male once). Where genotypes are missing, the frequency is slightly low (by at most 6.2%, 0.055 in absolute terms, in GTEx_v10 adipose). Where the metadata undercounts the genotyped samples, it is too high: the frequency is null where `ac` reaches 2 × `sample_size`. The minor allele frequency `maf` is not used ([opentargets/issues#4683](https://github.com/opentargets/issues/issues/4683)).
- A credible set whose lead has a p-value above 1e-3 (`lead_pvalue_threshold`) is dropped. The v3 run flagged them `Subsignificant p-value` instead.

## Contents of the run 20261117T103813

| Study type | Studies | Credible sets |
|---|---:|---:|
| `eqtl` | 1,733,667 | 2,138,792 |
| `sqtl` | 1,250,077 | 1,436,388 |
| `tuqtl` | 555,286 | 630,225 |
| `sceqtl` | 171,346 | 194,925 |
| `pqtl` | 801 | 1,624 |
| Total | 3,711,177 | 4,401,954 |

- 98,043,317 credible set variants: 1 to 4,931 per credible set, median 8.
- 58 projects (`projectId`), including `GTEx_v10` (1,949,884 studies), `IBDverse` (65,216) and `MAGE` (60,243).
- The study index has the same studies and values as the v3 run `20261029T162303`, except `biosampleFromSourceId` of the 11 `Natri_2024` studies: the string `NaN` in v3, null here. `pubmedId` is a whole number, and null for the 100,256 studies of `AFR_LCL`, which has no PubMed id in the eQTL Catalogue metadata.
- `id` of `credible_set` and (`credibleSetId`, `variantId`) of `credible_set_variant` are unique. Every credible set has a study and variants, and every variant has a credible set. 241 studies have no credible set; every v3 credible set of these studies has a lead p-value above 1e-3.
- 4,401,954 credible sets is the number of v3 credible sets with a lead p-value of at most 1e-3. 38 credible sets in each direction cross the threshold, because v3 and v4 break ties of the top posterior probability differently.
- Chromosomes `1` to `22` and `X`. `locusStart` is set for every credible set, and is 1 for 121,931.
- The posterior probabilities of a credible set sum to 0.95 to 1 for 4,398,811 sets, and to more than 1 for none. 3,143 sets sum to less than 0.95, 2,100 to less than 0.94, mostly in QTS000055 (1,260) and QTS000044 (710), where members with non-ATGC alleles were removed.
- `effectAlleleFrequency` is set for 96,931,465 of the 96,931,541 variants on chromosomes 1 to 22, between 0.0048 and 0.99983, median 0.274, and null for the 1,111,776 on chrX. It is null for 76 variants of the 3 `INTERVAL_RNA_WGS` datasets (QTD001003 to QTD001005), whose `ac` reaches 2 × `sample_size`: their metadata `sample_size` is 2,892, while their summary statistics count up to 2,917 samples (`an` up to 5,834, [eQTL-Catalogue/eQTL-Catalogue-resources#65](https://github.com/eQTL-Catalogue/eQTL-Catalogue-resources/issues/65)). The other frequencies of these datasets are about 0.9% too high.
- Against the run `20261117T083511` (gentropy `v4.0.0-dev.14`), only these 76 frequencies change, from 1 (10) or above 1 (66, at most 1.0035) to null; the study index, the credible sets and every other variant value are the same, but for `logBayesFactor` of 1,815,716 credible sets, which differs by at most 1.2e-12.
- Against the v4 run `20261105T130431`, the study index, the credible sets and the variants are the same, but for `effectAlleleFrequency` (null there) and `logBayesFactor` of 1,816,777 credible sets, which differs by at most 1e-12.
- Against the v3 run `20261029T162303`, the 4,401,916 credible sets in both runs have the same members, p-values, `reportedBeta` and `reportedStandardError`; posterior probabilities differ by at most 3e-8. 529,094 of them have a different lead of the same top posterior probability: v4 breaks the tie by variant id.
- Against the v4 run `20261030T164426`, only the 121,931 credible sets with a negative region start change, `locusStart` (null to 1) and `id`, besides `effectAlleleFrequency`.
- `conditionalBeta` and `conditionalStandardError` are null for 1.2% of variants, those of the 12 datasets with no full SuSiE posterior.
- Null in every row: `confidence` and `purityMeanR2` of `credible_set`; `sampleSize`, `r2Overall` and the rescaled statistics of `credible_set_variant`. `qualityControls` is empty.

## Outputs

### study_index

One row per dataset and molecular trait.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | `<study>_<dataset>_<dataset label>_<molecular trait>` in lower case, e.g. `qts000044_qtd000764_mage_ge_lcl_ensg00000273841`. |
| `projectId` | `string` | Study label of the eQTL Catalogue, e.g. `GTEx_v10`, `MAGE`, `BLUEPRINT`. |
| `studyType` | `string` | `eqtl`, `sqtl`, `tuqtl`, `sceqtl` or `pqtl`. |
| `traitFromSource` | `string` | Molecular trait id of the eQTL Catalogue: gene, exon, transcript, intron or event. |
| `geneId` | `string` | Ensembl gene id of the molecular trait. |
| `biosampleFromSourceId` | `string` | Tissue or cell type ontology term (UBERON, CL, EFO). |
| `condition` | `string` | Condition of the samples, e.g. `naive`, `sepsis`, `LPS_24h`. |
| `pubmedId` | `string` | PubMed id of the study; null for `AFR_LCL`. |
| `nSamples` | `integer` | Sample size of the dataset. |
| `summarystatsLocation` | `string` | eQTL Catalogue FTP folder of the summary statistics of the dataset. They are not ingested. |
| `hasSumstats` | `boolean` | `true`. |

### credible_set

One row per credible set, in the gentropy v4 [`CredibleSet`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.15/src/gentropy/dataset/credible_set.py) schema.

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Identifier of the credible set: an MD5 hash of `studyId`, `chromosome`, `locusStart`, `locusEnd`, `leadVariantId` and `fineMappingMethod`. |
| `chromosome` | `string` | Chromosome of the region. |
| `locusStart` | `integer` | Start of the fine-mapped region on GRCh38; 1 when the region starts before the chromosome. |
| `locusEnd` | `integer` | End of the fine-mapped region on GRCh38. |
| `studyId` | `string` | Study of the credible set, as in `study_index`. |
| `studyType` | `string` | As in `study_index`. |
| `leadVariantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `fineMappingMethod` | `string` | `SuSiE`. |
| `logBayesFactor` | `double` | Log Bayes factor of the credible set. |
| `confidence` | `byte` | Null. |
| `purityMeanR2` | `double` | Null. |
| `purityMinR2` | `double` | Smallest r2 between the members, from the source (`cs_min_r2`). |
| `qualityControls` | `array<string>` | Empty. |

### credible_set_variant

One row per member of a credible set, in the gentropy v4 [`CredibleSetVariant`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.15/src/gentropy/dataset/credible_set.py) schema.

| Column | Type | Description |
|---|---|---|
| `credibleSetId` | `string` | `id` of the credible set. |
| `variantId` | `string` | Member, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `pValueMantissa` | `float` | Mantissa of the marginal p-value (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the marginal p-value. |
| `effectAlleleFrequency` | `float` | Frequency of the ALT (effect) allele, `ac / (2 × sample_size)`, on chromosomes 1 to 22; null on chrX, and where `ac` reaches 2 × `sample_size`. |
| `sampleSize` | `integer` | Null: the source gives no sample size per variant; the dataset `sample_size` is only an upper bound. |
| `reportedBeta` | `double` | Marginal effect size reported by the study. |
| `reportedStandardError` | `double` | Standard error of `reportedBeta`. |
| `posteriorProbability` | `float` | Posterior probability of the member in the credible set. |
| `logBayesFactor` | `float` | Log Bayes factor of the member for the SuSiE component of the credible set. |
| `r2Overall` | `float` | Null. |
| `conditionalBeta` | `float` | SuSiE posterior mean of the effect of the credible set, unscaled; null for the 12 datasets with no full SuSiE posterior. |
| `conditionalStandardError` | `float` | SuSiE posterior standard deviation of that effect, unscaled; null as `conditionalBeta`. |
| `rescaledAbsBeta` | `float` | Null. |
| `rescaledStandardError` | `float` | Null. |
| `rescaledDirection` | `byte` | Null. |
| `qualityControls` | `array<string>` | Empty. |

## Changelog

### 2026-11-17

- feat: rerun r8 on gentropy v4.0.0-dev.14 into `gs://eqtl_catalogue_data/r8/prod/20261117T083511/`, with `effectAlleleFrequency` set to `ac / (2 × sample_size)` on chromosomes 1 to 22 ([opentargets/gentropy#1349](https://github.com/opentargets/gentropy/pull/1349), [opentargets/issues#4683](https://github.com/opentargets/issues/issues/4683)), so PTS can rescale the effects.
- fix: rerun r8 on gentropy v4.0.0-dev.15 into `gs://eqtl_catalogue_data/r8/prod/20261117T103813/`. The run `20261117T083511` gave 76 `INTERVAL_RNA_WGS` variants a frequency of 1 or above, as the metadata undercounts the samples of these datasets ([eQTL-Catalogue/eQTL-Catalogue-resources#65](https://github.com/eQTL-Catalogue/eQTL-Catalogue-resources/issues/65)); the frequency is now null where `ac` reaches 2 × `sample_size` ([opentargets/gentropy#1353](https://github.com/opentargets/gentropy/pull/1353)). PIS stages this run.

### 2026-11-05

- fix: rerun r8 on gentropy v4.0.0-rc.2 into `gs://eqtl_catalogue_data/r8/prod/20261105T130431/`. A negative region start is parsed as a `locusStart` of 1, so 121,931 credible sets get a start and a new `id` ([opentargets/issues#4652](https://github.com/opentargets/issues/issues/4652)). The duplicate QTS000038 credible sets are no longer written, so the run needs no patch.

### 2026-10-30

- feat: rerun r8 in the gentropy v4 schema with the staging DAG into `gs://eqtl_catalogue_data/r8/prod/20261030T153207/`, with the conditional statistics from the full SuSiE posterior. Credible sets with a lead p-value above 1e-3 are dropped instead of flagged.
- fix: patch the duplicate QTS000038 credible sets into `gs://eqtl_catalogue_data/r8/prod/20261030T164426/`.
- docs: 58 projects, not 59 as written on 2026-10-29.

### 2026-10-29

- fix: patch `pubmedId` of the r8 study index into `gs://eqtl_catalogue_data/r8/prod/20261029T162303/`, which the releases read.
- chore: copy the r8 run of 2026-08-22 from `gs://eqtl_catalogue_data/r8/` to `gs://eqtl_catalogue_data/r8/prod/20260822T223458/` (`config/`, `docs/`, `output/`). The public release now reads r8 instead of r7, and no longer reads GTEx v10, IBDverse and MAGE on their own; the PPP release reads the same run. The r7 outputs and the other old folders at the root of the bucket are described in the README of earlier commits.

### 2025-02-05

- [fix: reclassify eqtl catalogue sc datasets #894](https://github.com/opentargets/gentropy/pull/894)
- [feat(qtls): flagging trans QTL credible sets #973](https://github.com/opentargets/gentropy/pull/973)
- [chore: removing symbols from QTL study identifiers #971](https://github.com/opentargets/gentropy/pull/971)
- [refactor(eqtl catalogue): update dag structure for r8 release](https://github.com/opentargets/issues/issues/4421)
