# GWAS Catalog top hits

This document was updated on 2026-10-26.

```sh
gs://gwas_catalog_top_hits/prod/20260828T165117/config/gwas_catalog_top_hits.yaml   # DAG config of the run
gs://gwas_catalog_top_hits/prod/20260828T165117/docs/
gs://gwas_catalog_top_hits/prod/20260828T165117/output/credible_sets/
gs://gwas_catalog_top_hits/prod/20260828T165117/output/study_index/
gs://gwas_catalog_top_hits/prod/20260828T165117/output/study_locus_ld_clumped/
gs://gwas_catalog_top_hits/prod/20260828T165117/output/study_locus_window_based_clumped/
```

Data contained in the `top_hits` corresponds to the [GWAS Catalog top associations](https://www.ebi.ac.uk/gwas/docs/methods/criteria#:~:text=statistics%20here.-,Top%20associations,-We%20also%20report)

Data is produced by **gwas_catalog_top_hits DAG**. The dag configuration and topology can be found in `config/gwas_catalog_top_hits.yaml` of the run folder (`gs://gwas_catalog_top_hits/prod/20260828T165117/config/`). It runs gentropy [`v3.4.0-rc.3`](https://github.com/opentargets/gentropy/tree/v3.4.0-rc.3) (`GENTROPY_REF`), which the links below refer to.

The run read:

```sh
gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_download_studies.tsv
gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_download_ancestries.tsv
gs://gwas_catalog_inputs/gentroutils/20260802/gwas_catalog_associations_ontology_annotated.tsv
gs://gnomad_data_2/v4.1/variant_index/
gs://gnomad_data_2/v2.1.1/ld_index/
```

The three GWAS Catalog files are synced from the [EBI FTP](https://ftp.ebi.ac.uk/pub/databases/gwas/releases/latest/) by [gentroutils](https://github.com/opentargets/gentroutils), outside this DAG: studies (`gwas_catalog_download_studies.tsv`), their ancestries (`gwas_catalog_download_ancestries.tsv`) and the curated associations with their ontology annotations (`gwas_catalog_associations_ontology_annotated.tsv`). Each sync is written to a folder named after the date of the GWAS Catalog release, `gentroutils/<YYYYMMDD>/`.
DAG contains 3 steps:

![gwas_catalog_top_hits](gwas_catalog_top_hits.svg)

1. gwas_catalog_top_hit_ingestion
2. ld_based_clumping
3. pics

## gwas_catalog_top_hit_ingestion

The step ([`gwas_catalog_top_hits.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/gwas_catalog_top_hits.py)) runs on the dataproc cluster that is used to generate the [study_index dataset](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/dataset/study_index.py) and perform [window_based_clumping](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/method/window_based_clumping.py) on top associations. The datasets are saved under `gs://gwas_catalog_top_hits/prod/20260828T165117/output/study_index/` and `gs://gwas_catalog_top_hits/prod/20260828T165117/output/study_locus_window_based_clumped/`.

## ld_based_clumping

The step ([`ld_based_clumping.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/ld_based_clumping.py)) that performs [ld_clumping](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/method/clump.py) on the results from the previous step. The results from this step are saved under `gs://gwas_catalog_top_hits/prod/20260828T165117/output/study_locus_ld_clumped/`.

## pics

The step ([`pics.py`](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/pics.py)) that performs [PICS finemapping](https://github.com/opentargets/gentropy/blob/v3.4.0-rc.3/src/gentropy/method/pics.py) based on the ld clumped study locus dataset to obtain the dataset containing credible sets. This dataset is stored under `gs://gwas_catalog_top_hits/prod/20260828T165117/output/credible_sets/`

## Outputs

Schemas as read from the parquet files of the run. Nested columns show their `array<struct<...>>` types.

### study_index

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | GWAS Catalog study accession (`GCST…`). |
| `projectId` | `string` | Source of the study: `GCST` |
| `studyType` | `string` | Type of the study: `gwas`. |
| `pubmedId` | `string` | PubMed id of the publication. |
| `publicationFirstAuthor` | `string` | First author of the publication. |
| `publicationDate` | `string` | Publication date. |
| `publicationJournal` | `string` | Journal of the publication. |
| `publicationTitle` | `string` | Title of the publication. |
| `traitFromSource` | `string` | Trait as reported by the GWAS Catalog. |
| `initialSampleSize` | `string` | Discovery sample as reported, in free text. |
| `traitFromSourceMappedIds` | `array<string>` | Ontology (EFO) terms the trait is mapped to. Populated by GWAS Catalog. |
| `backgroundTraitFromSourceMappedIds` | `array<string>` | Ontology (EFO) terms of the background trait shared by all samples, e.g. a disease the cohort has. |
| `cohorts` | `array<string>` | Cohorts of the study, as reported. |
| `discoverySamples` | `array<struct<ancestry: string, sampleSize: integer>>` | Ancestry and size of each discovery sample. |
| `ldPopulationStructure` | `array<struct<ldPopulation: string, relativeSampleSize: double>>` | LD populations of the discovery samples and their share of the sample; the largest one is the LD reference of the study. |
| `replicationSamples` | `array<struct<ancestry: string, sampleSize: integer>>` | Ancestry and size of each replication sample. |
| `nCases` | `integer` | Number of cases. |
| `nControls` | `integer` | Number of controls. |
| `nSamples` | `integer` | Number of samples. |
| `qualityControls` | `array<string>` | Quality control flags of the study. |

### study_locus_window_based_clumped

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | GWAS Catalog study accession (`GCST…`) of the association, as in `study_index`. |
| `studyLocusId` | `string` | Identifier of the study locus: an md5 hash of `studyId` and `variantId`. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38, mapped to gnomAD v4.1. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `pValueMantissa` | `float` | Mantissa of the reported p-value (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the reported p-value. |
| `qualityControls` | `array<string>` | Quality control flags of the study locus. |
| `beta` | `double` | Effect size harmonised to the alternate allele of `variantId`; reported odds ratios are converted to beta. |
| `standardError` | `double` | Standard error of `beta`, derived from the effect size and the p-value, or from the odds ratio confidence interval. |

### study_locus_ld_clumped

| Column | Type | Description |
|---|---|---|
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38, from the gnomAD v4.1 variant index. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `studyId` | `string` | GWAS Catalog study accession (`GCST…`) of the association, as in `study_index`. |
| `studyLocusId` | `string` | Identifier of the study locus: a hash of `studyId` and `variantId`. |
| `studyType` | `string` | Empty in this release. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `region` | `string` | Empty in this release. |
| `beta` | `double` | Effect size harmonised to the alternate allele of `variantId`; reported odds ratios are converted to beta. |
| `zScore` | `double` | Empty in this release. |
| `pValueMantissa` | `float` | Mantissa of the reported p-value (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the reported p-value. |
| `effectAlleleFrequencyFromSource` | `float` | Empty in this release. |
| `standardError` | `double` | Standard error of `beta`, derived from the effect size and the p-value, or from the odds ratio confidence interval. |
| `subStudyDescription` | `string` | Empty in this release. |
| `qualityControls` | `array<string>` | Quality control flags of the study locus. |
| `finemappingMethod` | `string` | Empty until `pics`. |
| `credibleSetIndex` | `integer` | Empty in this release. |
| `credibleSetlog10BF` | `double` | Empty in this release. |
| `purityMeanR2` | `double` | Empty in this release. |
| `purityMinR2` | `double` | Empty in this release. |
| `locusStart` | `integer` | Empty in this release. |
| `locusEnd` | `integer` | Empty in this release. |
| `sampleSize` | `integer` | Empty in this release. |
| `locus` | `array<struct<is95CredibleSet: boolean, is99CredibleSet: boolean, logBF: double, posteriorProbability: double, variantId: string, pValueMantissa: float, pValueExponent: integer, beta: double, standardError: double, r2Overall: double>>` | Empty until `pics`. |
| `confidence` | `string` | Empty in this release. |
| `isTransQtl` | `boolean` | Empty in this release. |
| `ldSet` | `array<struct<tagVariantId: string, r2Overall: double>>` | Variants in LD with the lead (`tagVariantId`) and their r² (`r2Overall`) in the major LD population of the study; only r² of at least 0.5. |

### credible_sets

| Column | Type | Description |
|---|---|---|
| `studyLocusId` | `string` | Identifier of the study locus: a hash of `studyId` and `variantId`. |
| `studyType` | `string` | Empty in this release. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38, from the gnomAD v4.1 variant index. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `region` | `string` | Empty in this release. |
| `studyId` | `string` | GWAS Catalog study accession (`GCST…`) of the association, as in `study_index`. |
| `beta` | `double` | Effect size harmonised to the alternate allele of `variantId`; reported odds ratios are converted to beta. |
| `zScore` | `double` | Empty in this release. |
| `pValueMantissa` | `float` | Mantissa of the reported p-value (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the reported p-value. |
| `effectAlleleFrequencyFromSource` | `float` | Empty in this release. |
| `standardError` | `double` | Standard error of `beta`, derived from the effect size and the p-value, or from the odds ratio confidence interval. |
| `subStudyDescription` | `string` | Empty in this release. |
| `qualityControls` | `array<string>` | Quality control flags of the study locus. |
| `finemappingMethod` | `string` | Fine-mapping method: `PICS`. |
| `credibleSetIndex` | `integer` | Empty in this release. |
| `credibleSetlog10BF` | `double` | Empty in this release. |
| `purityMeanR2` | `double` | Empty in this release. |
| `purityMinR2` | `double` | Empty in this release. |
| `locusStart` | `integer` | Empty in this release. |
| `locusEnd` | `integer` | Empty in this release. |
| `sampleSize` | `integer` | Empty in this release. |
| `ldSet` | `array<struct<tagVariantId: string, r2Overall: double>>` | Variants in LD with the lead (`tagVariantId`) and their r² (`r2Overall`) in the major LD population of the study; only r² of at least 0.5. |
| `confidence` | `string` | Empty in this release. |
| `isTransQtl` | `boolean` | Empty in this release. |
| `locus` | `array<struct<variantId: string, r2Overall: double, posteriorProbability: double, standardError: double, pValueMantissa: float, pValueExponent: integer, beta: double, is95CredibleSet: boolean, is99CredibleSet: boolean>>` | Variants of the credible set: `variantId`, its r² with the lead (`r2Overall`), its `posteriorProbability`, and whether it is in the 95% and 99% credible sets (`is95CredibleSet`, `is99CredibleSet`). |
