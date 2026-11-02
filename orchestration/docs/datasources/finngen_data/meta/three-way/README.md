# FinnGen, UK Biobank and MVP meta-analysis

This document was updated on 2026-11-01.

```sh
gs://finngen_data/meta/three-way/prod/20251027T140104/config/finngen_ukb_mvp_meta.yaml # DAG config of the run
gs://finngen_data/meta/three-way/prod/20251027T140104/docs/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/raw_summary_statistics/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/harmonised_summary_statistics/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/harmonised_summary_statistics_qc/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/study_index/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/study_locus_window_based_clumped/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/study_locus_ld_clumped/
gs://finngen_data/meta/three-way/prod/20251027T140104/output/credible_sets/
```

Data contained in `meta/three-way` corresponds to the three-way meta-analysis of [FinnGen R12, the Million Veteran Program (MVP) and the Pan-UK Biobank](https://mvp-ukbb.finngen.fi/about), published by FinnGen in `gs://finngen-public-data-r12/meta_analysis/mvp_ukbb/`. The meta-analysis covers 330 phenotypes. The processing is described in [opentargets/issues#3474](https://github.com/opentargets/issues/issues/3474).

Data is produced by **finngen_ukb_mvp_meta DAG**. The DAG configuration and topology are in `config/finngen_ukb_mvp_meta.yaml` of the run folder (`gs://finngen_data/meta/three-way/prod/20251027T140104/config/`). The run uses gentropy [`v2.4.2-internal.7`](https://github.com/opentargets/gentropy/tree/v2.4.2-internal.7) (`GENTROPY_REF`); the links below refer to that tag.

The run folder is named after the write time of the first output of the run, the first `raw_summary_statistics` partition (2025-10-27 14:01:04 UTC). The run outputs were written to `gs://finngen_ukb_mvp_meta_data/` and copied to the run folder on 2026-11-01.

Inputs of the run:

```sh
gs://finngen-public-data-r12/meta_analysis/mvp_ukbb/FinnGen_R12_MVP_UKBB_manifest.tsv
gs://finngen-public-data-r12/meta_analysis/mvp_ukbb/summary_stats/*_meta_out.tsv.gz
https://raw.githubusercontent.com/opentargets/curation/refs/tags/25.09/mappings/disease/manual_string.tsv
gs://gnomad_data_2/v4.1/variant_index/
gs://gnomad_data_2/v2.1.1/ld_index/
```

The manifest lists the 330 phenotypes with their case and control counts per cohort, and the location of the summary statistics of each phenotype in its `path_bucket` column. The `manual_string.tsv` file of the Open Targets curation repository, at tag `25.09`, maps FinnGen phenotypes to EFO terms. The gnomAD v4.1 variant index is the reference for allele orientation, and the gnomAD v2.1.1 LD index is the LD reference of the LD-based clumping.

The DAG has 4 steps, run in sequence on one Dataproc cluster:

![finngen_ukb_mvp_meta](finngen_ukb_mvp_meta.svg)

1. harmonisation
2. window_based_clumping
3. ld_based_clumping
4. pics

## harmonisation

The step ([`finngen_ukb_mvp_meta.py`](https://github.com/opentargets/gentropy/blob/v2.4.2-internal.7/src/gentropy/finngen_ukb_mvp_meta.py)) writes four outputs.

The study index holds one study per manifest row ([`study_index.py`](https://github.com/opentargets/gentropy/blob/v2.4.2-internal.7/src/gentropy/datasource/finngen_meta/study_index.py)). The study identifier is the FinnGen phenotype code prefixed with `FINNGEN_R12_UKB_MVP_META_`. The ontology terms of a study come from the curation rows of FinnGen R12 whose phenotype name matches the study.

In the first step, the source summary statistics are reformatted to a parquet dataset partitioned by `studyId`, named `raw_summary_statistics`.

The harmonised summary statistics are built from the raw summary statistics:

- The effect is the inverse-variance meta-analysis effect over all cohorts (`all_inv_var_meta_beta`, `all_inv_var_meta_sebeta`, `all_inv_var_meta_mlogp`). Variants without these values are removed.
- A variant is kept when it is present in at least two of the three biobanks (FinnGen, MVP, UK Biobank).
- A variant is removed when its imputation score (`r2`) is below 0.8 in any MVP cohort. A missing score passes the filter.
- A variant found in the gnomAD v4.1 variant index takes the gnomAD allele orientation; the effect and the allele frequencies are flipped when the alleles are swapped. A variant not found in gnomAD keeps the source orientation.
- The effect allele frequency is the sample-size-weighted mean of the cohort allele frequencies.
- A variant is removed when its minor allele count is below 20 in any cohort.

The summary statistics QC holds one row per study with the QC metrics of its harmonised summary statistics ([`summary_statistics_qc.py`](https://github.com/opentargets/gentropy/blob/v2.4.2-internal.7/src/gentropy/dataset/summary_statistics_qc.py)). The step then rewrites the study index with these metrics in `sumstatQCValues`.

## window_based_clumping

The step ([`window_based_clumping.py`](https://github.com/opentargets/gentropy/blob/v2.4.2-internal.7/src/gentropy/window_based_clumping.py)) selects the lead variants of the harmonised summary statistics: variants with a p-value of at most 1e-8, with one lead per 500 kb window. The output is `output/study_locus_window_based_clumped/`: 15,944 loci from 269 studies.

## ld_based_clumping

The step ([`ld_based_clumping.py`](https://github.com/opentargets/gentropy/blob/v2.4.2-internal.7/src/gentropy/ld_based_clumping.py)) annotates each locus with the variants in LD with its lead (r² of at least 0.5) from the gnomAD v2.1.1 LD index. The r² is the r² of the major population of the study, the population with the largest `relativeSampleSize` in `ldPopulationStructure` (`nfe` for 317 studies and `fin` for 13 studies). A lead with no LD information is flagged `Variant not found in LD reference` and keeps itself as the only variant of its LD set. A lead in high LD with a more significant lead of the same study is flagged `Explained by a more significant variant in high LD`, and its LD set is emptied; no locus is removed. The output is `output/study_locus_ld_clumped/`: 15,944 loci.

## pics

The step ([`pics.py`](https://github.com/opentargets/gentropy/blob/v2.4.2-internal.7/src/gentropy/pics.py)) fine-maps each LD-annotated locus with [PICS](https://opentargets.github.io/gentropy/python_api/methods/pics/) and keeps the variants of the 99% credible set in `locus`. The output is `output/credible_sets/`: 15,944 credible sets from 269 studies. Every credible set is flagged `Study locus finemapped without in-sample LD reference`. The 566 credible sets flagged `Explained by a more significant variant in high LD` have an empty `ldSet` and a null `locus`. The 362 credible sets flagged `Variant not found in LD reference` hold the lead as the only variant of `locus`. Every study of the credible sets is in `study_index`.

## Outputs

### raw_summary_statistics

One row per variant of a source file, partitioned by `studyId`: 330 partitions and 1.18 TiB. The columns are the 67 columns of the source files, for example `#CHR`, `POS`, `REF`, `ALT`, the per-cohort statistics (`fg_*`, `MVP_EUR_*`, `MVP_AFR_*`, `MVP_HIS_*`, `ukbb_*`) and the meta-analysis statistics (`all_inv_var_meta_*` and the leave-one-out `leave_*_inv_var_meta_*`).

### harmonised_summary_statistics

One row per variant and study: 3,853,054,848 rows from 330 studies.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | Study, as in `study_index`. |
| `variantId` | `string` | `<chromosome>_<position>_<ref>_<alt>` on GRCh38, in the gnomAD orientation when the variant is in gnomAD. |
| `chromosome` | `string` | Chromosome. |
| `position` | `integer` | Position on GRCh38. |
| `beta` | `double` | Meta-analysis effect over all cohorts. |
| `sampleSize` | `integer` | Number of samples of the study. |
| `pValueMantissa` | `float` | Mantissa of the meta-analysis p-value (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the meta-analysis p-value. |
| `effectAlleleFrequencyFromSource` | `float` | Sample-size-weighted mean of the cohort allele frequencies. |
| `standardError` | `double` | Standard error of the meta-analysis effect. |

### harmonised_summary_statistics_qc

One row per study: 330 rows.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | Study, as in `study_index`. |
| `mean_beta` | `double` | Mean effect over all variants. |
| `mean_diff_pz` | `double` | Mean difference between the -log10 p-value of the source and the -log10 p-value derived from the z-score (`beta` / `standardError`). |
| `se_diff_pz` | `double` | Standard error of that difference. |
| `gc_lambda` | `double` | Genomic inflation factor. |
| `n_variants` | `long` | Number of variants. |
| `n_variants_sig` | `long` | Number of variants with a p-value below 1e-8. |

### study_index

One row per phenotype: 330 studies. The table lists the populated columns; the other `StudyIndex` columns are null.

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | `FINNGEN_R12_UKB_MVP_META_<phenocode>`. |
| `projectId` | `string` | `FINNGEN_R12_UKB_MVP_META`. |
| `studyType` | `string` | `gwas`. |
| `traitFromSource` | `string` | Phenotype name, as in the manifest. |
| `traitFromSourceMappedIds` | `array<string>` | Ontology terms of the phenotype, from the curation file. |
| `publicationDate` | `string` | `2024-11-01`. |
| `initialSampleSize` | `string` | `1,550,147 (MVP: nEUR=449,042, nAFR=121,177, nAMR=59,048; FinnGenR12: nNFE=500,349; pan-UKBB-EUR: nEUR=420,531)`. |
| `nCases` | `integer` | Number of cases over all cohorts. |
| `nControls` | `integer` | Number of controls over all cohorts. |
| `nSamples` | `integer` | `nCases` + `nControls`. |
| `cohorts` | `array<string>` | `[MVP, FinnGen, pan-UKBB-EUR]`. |
| `discoverySamples` | `array<struct<sampleSize: integer, ancestry: string>>` | Samples per ancestry: `Finnish` (FinnGen), `European` (MVP EUR and UK Biobank), `African` (MVP AFR) and `Admixed American` (MVP HIS). |
| `ldPopulationStructure` | `array<struct<ldPopulation: string, relativeSampleSize: double>>` | `fin`, `nfe`, `afr` and `amr`, weighted by the sample size of each ancestry. |
| `qualityControls` | `array<string>` | Empty for every study. |
| `summarystatsLocation` | `string` | Source summary statistics of the phenotype, `gs://finngen-public-data-r12/meta_analysis/mvp_ukbb/summary_stats/<phenocode>_meta_out.tsv.gz`. |
| `sumstatQCValues` | `array<struct<QCCheckName: string, QCCheckValue: float>>` | The metrics of `harmonised_summary_statistics_qc`. |
| `hasSumstats` | `boolean` | `true`. |

### study_locus_window_based_clumped

One row per lead variant: 15,944 loci from 269 studies, with the columns of `harmonised_summary_statistics`, `studyLocusId` and `qualityControls`.

### study_locus_ld_clumped

One row per locus: 15,944 loci, with the columns of `study_locus_window_based_clumped` and `ldSet`, the variants in LD with the lead and their r² (`tagVariantId`, `r2Overall`). The other `StudyLocus` columns, `locus` included, are null.

### credible_sets

One row per credible set: 15,944 credible sets from 269 studies.

| Column | Type | Description |
|---|---|---|
| `studyLocusId` | `string` | Identifier of the credible set: a hash of `studyId`, `variantId` and `finemappingMethod`. |
| `studyId` | `string` | Study of the credible set, as in `study_index`. |
| `variantId` | `string` | Lead variant, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `chromosome` | `string` | Chromosome of the lead variant. |
| `position` | `integer` | Position of the lead variant on GRCh38. |
| `beta` | `double` | Meta-analysis effect of the lead. |
| `pValueMantissa` | `float` | Mantissa of the p-value of the lead (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the p-value of the lead. |
| `effectAlleleFrequencyFromSource` | `float` | Effect allele frequency of the lead. |
| `standardError` | `double` | Standard error of the lead. |
| `sampleSize` | `integer` | Number of samples of the study. |
| `finemappingMethod` | `string` | `PICS`. |
| `ldSet` | `array<struct<tagVariantId: string, r2Overall: double>>` | Variants in LD with the lead. |
| `locus` | `array<struct<variantId: string, r2Overall: double, posteriorProbability: double, standardError: double, pValueMantissa: float, pValueExponent: integer, beta: double, is95CredibleSet: boolean, is99CredibleSet: boolean>>` | Variants of the 99% credible set, from 1 to 3,402 per credible set, with their PICS posterior probability. Null for the 566 credible sets flagged `Explained by a more significant variant in high LD`. |
| `qualityControls` | `array<string>` | QC flags, see [pics](#pics). |

`studyType`, `zScore`, `region`, `credibleSetIndex`, `credibleSetlog10BF`, `purityMeanR2`, `purityMinR2`, `locusStart`, `locusEnd`, `subStudyDescription`, `confidence` and `isTransQtl` are part of the schema and null in every row.

## Changelog

- 2025-10-27: initial pipeline run, written to `gs://finngen_ukb_mvp_meta_data/`.
- 2026-11-01: the run outputs are copied to `gs://finngen_data/meta/three-way/prod/20251027T140104/output/`, with the DAG config under `config/` and these docs under `docs/`.
