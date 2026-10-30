# GWAS Catalog summary statistics PICS

This document was updated on 2026-10-29.

```sh
gs://gwas_catalog_sumstats_pics/prod/20260828T115527/config/gwas_catalog_sumstats_pics.yaml # DAG config of the run
gs://gwas_catalog_sumstats_pics/prod/20260828T115527/docs/
gs://gwas_catalog_sumstats_pics/prod/20260828T115527/output/study_index/
gs://gwas_catalog_sumstats_pics/prod/20260828T115527/output/study_locus_window_based_clumped/
gs://gwas_catalog_sumstats_pics/prod/20260828T115527/output/study_locus_ld_clumped/
gs://gwas_catalog_sumstats_pics/prod/20260828T115527/output/credible_sets/
```

Data contained in the `sumstats_pics` corresponds to the GWAS Catalog summary statistics harmonised to the Open Targets [SummaryStatistics dataset](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/dataset/summary_statistics.py), fine-mapped with PICS.

Data is produced by **gwas_catalog_sumstats_pics DAG**. The DAG configuration and topology are in `config/gwas_catalog_sumstats_pics.yaml` of the run folder (`gs://gwas_catalog_sumstats_pics/prod/20260828T115527/config/`). The run uses gentropy [`v3.4.0-rc.3`](https://github.com/opentargets/gentropy/tree/v3.4.0-rc.3) (`GENTROPY_REF`); the links below refer to that tag.

Inputs of the run:

```sh
gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_download_studies.tsv
gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_download_ancestries.tsv
gs://gwas_catalog_inputs/harmonised_summary_statistics/
gs://gnomad_data_2/v2.1.1/ld_index/
```

The two GWAS Catalog files are synced from the [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/) by [gentroutils](https://github.com/opentargets/gentroutils), outside this DAG: studies (`gwas_catalog_download_studies.tsv`) and their ancestries (`gwas_catalog_download_ancestries.tsv`). Each sync is written to a folder named after the date of the GWAS Catalog release, `gentroutils/<YYYYMMDD>/`. The harmonised summary statistics are the output of the `gwas_catalog_harmonisation` DAG, one folder per study (`harmonised_summary_statistics/<studyId>/`); see the [GWAS Catalog inputs](../README.md#gwas-catalog-inputs).

The DAG has 4 steps, run on one Dataproc cluster:

![gwas_catalog_sumstats_pics](gwas_catalog_sumstats_pics.svg)

1. gwas_catalog_study_index
2. window_based_clumping
3. ld_based_clumping
4. pics

## gwas_catalog_study_index

The step ([`gwas_catalog_study_index.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/gwas_catalog_study_index.py)) builds one study per GWAS Catalog study from the studies and ancestries files ([`study_index.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/datasource/gwas_catalog/study_index.py)). The study index covers every study of the GWAS Catalog release, with or without summary statistics, and carries no curation or summary statistics QC annotation. The output is `output/study_index/`.

## window_based_clumping

The step ([`window_based_clumping.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/window_based_clumping.py)) reads all harmonised summary statistics and keeps, per study, the variants with a p-value of at most 1e-8 that are the most significant within 500 kb ([`window_based_clumping.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/method/window_based_clumping.py)). Each kept variant is the lead of one study locus. The output is `output/study_locus_window_based_clumped/`.

## ld_based_clumping

The step ([`ld_based_clumping.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/ld_based_clumping.py)) adds to each study locus the variants in LD with its lead, from the gnomAD v2.1.1 LD index, in the major LD population of the study. A study locus whose lead is in LD with the lead of a more significant study locus of the same study is flagged `Explained by a more significant variant in high LD` and its LD set is emptied ([`clump.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/method/clump.py)). A study locus whose lead is missing from the LD index is flagged `Variant not found in LD reference`, and its LD set holds the lead only. The output is `output/study_locus_ld_clumped/`.

## pics

The step ([`pics.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/pics.py)) fine-maps each study locus with [PICS](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/method/pics.py), from the p-value of the lead and its LD set. Every study locus, flagged or not, gives one row, flagged `Study locus finemapped without in-sample LD reference`; a study locus with an empty LD set gives an empty credible set. The output is `output/credible_sets/`.

## Outputs

### study_index

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | GWAS Catalog study accession (`GCST…`). |
| `projectId` | `string` | Source of the study: `GCST`. |
| `studyType` | `string` | Type of the study: `gwas`. |
| `pubmedId` | `string` | PubMed id of the publication. |
| `publicationFirstAuthor` | `string` | First author of the publication. |
| `publicationDate` | `string` | Publication date. |
| `publicationJournal` | `string` | Journal of the publication. |
| `publicationTitle` | `string` | Title of the publication. |
| `traitFromSource` | `string` | Trait as reported by the GWAS Catalog. |
| `initialSampleSize` | `string` | Discovery sample as reported, in free text. |
| `traitFromSourceMappedIds` | `array<string>` | Ontology (EFO) terms the trait is mapped to by the GWAS Catalog. |
| `backgroundTraitFromSourceMappedIds` | `array<string>` | Ontology (EFO) terms of the background trait shared by all samples, e.g. a disease of the cohort. |
| `cohorts` | `array<string>` | Cohorts of the study, as reported. |
| `discoverySamples` | `array<struct<ancestry: string, sampleSize: integer>>` | Ancestry and size of each discovery sample. |
| `ldPopulationStructure` | `array<struct<ldPopulation: string, relativeSampleSize: double>>` | LD populations of the discovery samples and their share of the sample; the largest one is the LD population of the study. |
| `replicationSamples` | `array<struct<ancestry: string, sampleSize: integer>>` | Ancestry and size of each replication sample. |
| `nCases` | `integer` | Number of cases. |
| `nControls` | `integer` | Number of controls. |
| `nSamples` | `integer` | Number of samples. |

### study_locus_window_based_clumped

One row per lead.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | Study of the summary statistics, as in `study_index`. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `beta` | `double` | Effect size of the lead, harmonised to the alternate allele. |
| `sampleSize` | `integer` | Sample size of the lead, where reported. |
| `pValueMantissa` | `float` | Mantissa of the p-value (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the p-value. |
| `effectAlleleFrequencyFromSource` | `float` | Frequency of the effect allele, where reported. |
| `standardError` | `double` | Standard error of `beta`. |
| `studyLocusId` | `string` | Identifier of the study locus: a hash of `studyId` and `variantId`. |
| `qualityControls` | `array<string>` | Empty. |

### study_locus_ld_clumped

The rows of `study_locus_window_based_clumped`, with the LD flags of `ld_based_clumping`.

| Column | Type | Description |
|---|---|---|
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `studyId` | `string` | Study of the study locus, as in `study_index`. |
| `studyLocusId` | `string` | Identifier of the study locus: a hash of `studyId` and `variantId`. |
| `studyType` | `string` | Empty. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `region` | `string` | Empty. |
| `beta` | `double` | Effect size of the lead, harmonised to the alternate allele. |
| `zScore` | `double` | Empty. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the lead. |
| `pValueExponent` | `integer` | Exponent of the p-value of the lead. |
| `effectAlleleFrequencyFromSource` | `float` | Frequency of the effect allele of the lead, where reported. |
| `standardError` | `double` | Standard error of `beta`. |
| `subStudyDescription` | `string` | Empty. |
| `qualityControls` | `array<string>` | Quality control flags of the study locus. |
| `finemappingMethod` | `string` | Empty. |
| `credibleSetIndex` | `integer` | Empty. |
| `credibleSetlog10BF` | `double` | Empty. |
| `purityMeanR2` | `double` | Empty. |
| `purityMinR2` | `double` | Empty. |
| `locusStart` | `integer` | Empty. |
| `locusEnd` | `integer` | Empty. |
| `sampleSize` | `integer` | Sample size of the lead, where reported. |
| `locus` | `array<struct<is95CredibleSet: boolean, is99CredibleSet: boolean, logBF: double, posteriorProbability: double, variantId: string, pValueMantissa: float, pValueExponent: integer, beta: double, standardError: double, r2Overall: double>>` | Empty. |
| `confidence` | `string` | Empty. |
| `isTransQtl` | `boolean` | Empty. |
| `ldSet` | `array<struct<tagVariantId: string, r2Overall: double>>` | Variants in LD with the lead (`tagVariantId`) and their r² with it (`r2Overall`) in the major LD population of the study. Empty for a study locus flagged `Explained by a more significant variant in high LD`. |

### credible_sets

One row per study locus.

| Column | Type | Description |
|---|---|---|
| `studyLocusId` | `string` | Identifier of the study locus: a hash of `studyId` and `variantId`. |
| `studyType` | `string` | Empty. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `region` | `string` | Empty. |
| `studyId` | `string` | Study of the credible set, as in `study_index`. |
| `beta` | `double` | Effect size of the lead, harmonised to the alternate allele. |
| `zScore` | `double` | Empty. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the lead. |
| `pValueExponent` | `integer` | Exponent of the p-value of the lead. |
| `effectAlleleFrequencyFromSource` | `float` | Frequency of the effect allele of the lead, where reported. |
| `standardError` | `double` | Standard error of `beta`. |
| `subStudyDescription` | `string` | Empty. |
| `qualityControls` | `array<string>` | Quality control flags of the study locus, plus `Study locus finemapped without in-sample LD reference`. |
| `finemappingMethod` | `string` | Fine-mapping method: `PICS`. |
| `credibleSetIndex` | `integer` | Empty. |
| `credibleSetlog10BF` | `double` | Empty. |
| `purityMeanR2` | `double` | Empty. |
| `purityMinR2` | `double` | Empty. |
| `locusStart` | `integer` | Empty. |
| `locusEnd` | `integer` | Empty. |
| `sampleSize` | `integer` | Sample size of the lead, where reported. |
| `ldSet` | `array<struct<tagVariantId: string, r2Overall: double>>` | As in `study_locus_ld_clumped`. |
| `confidence` | `string` | Empty. |
| `isTransQtl` | `boolean` | Empty. |
| `locus` | `array<struct<variantId: string, r2Overall: double, posteriorProbability: double, standardError: double, pValueMantissa: float, pValueExponent: integer, beta: double, is95CredibleSet: boolean, is99CredibleSet: boolean>>` | Variants of the 99% credible set: `variantId`, its r² with the lead (`r2Overall`), its PICS `posteriorProbability`, and whether it is in the 95% and 99% credible sets. `pValueMantissa`, `pValueExponent`, `beta` and `standardError` are set for the lead only. Empty for a study locus with an empty `ldSet`. |
