# BigBrain

This document was updated on 2026-10-29.

```sh
gs://bigbrain_data/prod/20261029T125742/config/bigbrain_ingestion.yaml # DAG config of the run
gs://bigbrain_data/prod/20261029T125742/bin/      # gentropy CLI and cluster init script of the run
gs://bigbrain_data/prod/20261029T125742/docs/     # this file
gs://bigbrain_data/prod/20261029T125742/source/   # input files, copied from Zenodo
gs://bigbrain_data/prod/20261029T125742/output/eqtl/study/
gs://bigbrain_data/prod/20261029T125742/output/eqtl/credible_set/
gs://bigbrain_data/prod/20261029T125742/output/eqtl/credible_set_variant/
gs://bigbrain_data/prod/20261029T125742/output/sqtl/study/
gs://bigbrain_data/prod/20261029T125742/output/sqtl/credible_set/
gs://bigbrain_data/prod/20261029T125742/output/sqtl/credible_set_variant/
```

[BigBrain](https://zenodo.org/records/17226890) is a brain-tissue molecular QTL meta-analysis (Réal, Kailash et al. 2025, "Mapping genetic effects on splicing in ten thousand post-mortem brain samples reveals novel mediators of neurological disease risk"). Its published SuSiE fine-mapping of the European (EUR) release is ingested as credible sets. It was run by BigBrain with PolyFun + SuSiE (L=5, 1 Mb window around the feature) on Z-scores, with in-sample LD from 3,884 European donors. The credible sets are not fine-mapped again.

Data is produced by the **bigbrain_ingestion DAG**. The DAG configuration is in `config/bigbrain_ingestion.yaml` of the run folder. The run uses gentropy [`v4.0.0-dev.5`](https://github.com/opentargets/gentropy/tree/v4.0.0-dev.5) and the step [`bigbrain_finemapping_ingestion`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.5/src/gentropy/bigbrain_ingestion.py) ([gentropy#1321](https://github.com/opentargets/gentropy/pull/1321)). Each run writes to its own folder `gs://bigbrain_data/prod/<YYYYMMDDTHHMMSS>/`.

Inputs of the run:

```sh
gs://bigbrain_data/prod/20261029T125742/source/BigBrain_cis_eQTL_finemapping.txt.gz     # Zenodo record 17226890
gs://bigbrain_data/prod/20261029T125742/source/BigBrain_cis_sQTL_finemapping.txt.gz     # Zenodo record 17153730
gs://bigbrain_data/prod/20261029T125742/source/BigBrain_cis_sQTL_EUR_top_assoc.tsv.gz   # Zenodo record 17153730, junction-to-gene mapping
gs://open-targets-pipeline-runs/do/platform-2610-2/output/target/                    # target index, sQTL only
```

The DAG has 3 steps:

1. `download`: copies the three BigBrain files from [record 17226890](https://zenodo.org/records/17226890) and [record 17153730](https://zenodo.org/records/17153730) into `source/`.
2. `eqtl`: one study per gene. Writes `output/eqtl/`.
3. `sqtl`: one study per intron junction (Leafcutter). Writes `output/sqtl/`.

`eqtl` and `sqtl` run in parallel on one Dataproc cluster.

## Contents of the run 20261029T125742

| | Studies | Credible sets | Credible set variants |
|---|---:|---:|---:|
| eQTL | 16,691 | 42,082 | 296,754 |
| sQTL | 35,879 | 68,808 | 654,666 |

- Each `output/<qtl>/<dataset>/` is a folder of Parquet files.
- `credible_set` is keyed by `id`. `credible_set_variant` joins to it on `credibleSetId`, and `credible_set` joins to `study` on `studyId`.
- Every credible set has a study and at least one variant. `id` is unique.
- 3,121 sQTL studies have no `geneId`: the junction has no gene, or it has several genes and none could be chosen (see `study`).

## How the source is converted

From [`finemapping.py`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.5/src/gentropy/datasource/bigbrain/finemapping.py) and [`study_index.py`](https://github.com/opentargets/gentropy/blob/v4.0.0-dev.5/src/gentropy/datasource/bigbrain/study_index.py):

- The published credible sets are kept as they are. PolyFun reports the SuSiE 95% credible sets.
- `variantId` is `CHR_BP_A2_A1`, so the effect allele `A1` is the alternate allele.
- Rows with `Z = 0` are untested placeholder variants. Credible sets led by one (about 2%) are dropped, and other such rows are removed.
- `posteriorProbability` is the published PIP scaled to sum to 0.9501 per credible set. The published PIP is the probability of being causal in any SuSiE effect of the feature, so the raw sums range from 0.03 to 2.6. Scaling keeps the order of the members.
- The lead variant is the one with the top PIP, ties broken by `variantId`. `locusStart` and `locusEnd` are the positions of the first and last member.
- P-values are recomputed from `Z`.
- Bayes factors are not published, so they are null. Use these sets with PIP-based colocalisation (`coloc_pip_ecaviar`).
- Purity is not published. It is 1 for single-variant sets and null for others. SuSiE only reports sets with a minimum absolute correlation of 0.5, so all sets passed a purity filter at source. Credible set QC that drops null purity would drop most sets.
- `conditionalBeta` and `conditionalStandardError` are the SuSiE posterior mean and standard deviation (`BETA_MEAN`, `BETA_SD`), summed over all effects of the feature and on the standardised scale. Use them for the direction of effect only, not for effect size. For low-PIP members the sign can come from another effect. No marginal beta is published.
- sQTL: a junction annotated to several genes (about 10%) is resolved with the target index to the one gene that contains it, or else to the one protein-coding gene that contains it. Otherwise `geneId` is null.

## Outputs

### study

One row per gene (eQTL) or intron junction (sQTL).

| Column | Type | Description |
|---|---|---|
| `studyId` | `string` | `BigBrain_<eqtl\|sqtl>_EUR_<feature>`, e.g. `BigBrain_eqtl_EUR_ENSG00000173627.8`, `BigBrain_sqtl_EUR_chr1_54668937_54670799`. |
| `geneId` | `string` | Ensembl gene id without version. sQTL: null for 3,121 junctions. |
| `traitFromSource` | `string` | eQTL: versioned Ensembl gene id. sQTL: gene symbols of the junction from `top_assoc`, joined by `;`; null for 144 junctions. |
| `projectId` | `string` | `BigBrain`. |
| `studyType` | `string` | `eqtl` or `sqtl`. |
| `biosampleFromSourceId` | `string` | `UBERON_0000955` (brain). |
| `pubmedId` | `string` | `35058635`, the mmQTL meta-analysis method paper. |
| `publicationFirstAuthor` | `string` | `Réal A, Kailash BP`. |
| `publicationDate` | `string` | `2025`. |
| `publicationTitle` | `string` | Title of the BigBrain paper. |
| `initialSampleSize` | `string` | `10,725 donors (EUR)`. 10,725 is the sample count of the whole resource, not split by ancestry. |
| `nSamples` | `integer` | `10725`, as above. |
| `discoverySamples` | `array<struct<sampleSize: integer, ancestry: string>>` | `[{10725, European}]`. |
| `ldPopulationStructure` | `array<struct<ldPopulation: string, relativeSampleSize: double>>` | `[{nfe, 1.0}]`. |
| `cohorts` | `array<string>` | `[BigBrain]`. |
| `summarystatsLocation` | `string` | Zenodo URL of the full summary statistics of the QTL type. They are not ingested. |
| `hasSumstats` | `boolean` | `true`. |

### credible_set

One row per credible set.

| Column | Type | Description |
|---|---|---|
| `id` | `string` | Identifier of the credible set. |
| `studyId` | `string` | Study of the credible set, as in `study`. |
| `studyType` | `string` | `eqtl` or `sqtl`. |
| `chromosome` | `string` | Chromosome, `1` to `22`. |
| `locusStart` | `integer` | Position of the first member variant on GRCh38. |
| `locusEnd` | `integer` | Position of the last member variant on GRCh38. |
| `leadVariantId` | `string` | Member with the top PIP, `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `fineMappingMethod` | `string` | `SuSiE`. |
| `logBayesFactor` | `double` | Null. |
| `confidence` | `tinyint` | Null. |
| `purityMeanR2` | `double` | 1 for single-variant sets (eQTL 16,018, sQTL 25,912), null otherwise. |
| `purityMinR2` | `double` | As `purityMeanR2`. |
| `qualityControls` | `array<string>` | Empty. |

### credible_set_variant

One row per member variant of a credible set.

| Column | Type | Description |
|---|---|---|
| `credibleSetId` | `string` | `id` of the credible set. |
| `variantId` | `string` | `<chromosome>_<position>_<ref>_<alt>` on GRCh38. |
| `posteriorProbability` | `float` | Published PIP, scaled to sum to 0.9501 per credible set. |
| `pValueMantissa` | `float` | Mantissa of the p-value from `Z` (p = mantissa × 10^exponent). |
| `pValueExponent` | `integer` | Exponent of the p-value. |
| `sampleSize` | `integer` | `N` of the association test at the variant. |
| `conditionalBeta` | `float` | SuSiE posterior mean (`BETA_MEAN`), summed over effects, standardised scale. Direction only. |
| `conditionalStandardError` | `float` | SuSiE posterior standard deviation (`BETA_SD`), as above. |
| `effectAlleleFrequency` | `float` | Null. |
| `reportedBeta` | `double` | Null. |
| `reportedStandardError` | `double` | Null. |
| `logBayesFactor` | `float` | Null. |
| `r2Overall` | `float` | Null. |
| `rescaledAbsBeta` | `float` | Null. |
| `rescaledStandardError` | `float` | Null. |
| `rescaledDirection` | `tinyint` | Null. |
| `qualityControls` | `array<string>` | Empty. |

## Other runs

- `gs://bigbrain_data/prod/20261015T103255/`: an earlier run from the gentropy#1321 branch (commit `777abba9`), with the eQTL and sQTL credible sets together in `output/credible_set/` and `output/study_index/`. Same counts. See its `docs/README.md`.

## Changelog

### 2026-10-29

- feat: add the `bigbrain_ingestion` DAG, a StagingDagConfig DAG on gentropy `v4.0.0-dev.5`.
- docs: describe the outputs of the run `20261029T125742`; the same file is in `docs/` of the run folder.
