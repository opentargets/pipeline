# eQTLGen single-cell eQTLs and co-eQTLs

This document was updated on 2026-11-03.

The eQTLGen consortium single-cell data has two datasets, fine-mapped with SuSiE by eQTLGen:

* **`single_cell_eqtl`**: cis-eQTLs of 6 blood cell types (B, CD4 T, CD8 T, DC, monocytes, NK). One study per cell type and gene: `eQTLGen_<cellType>_<ENSG>`.
* **`coeqtl`**: co-expression QTLs of 5 cell types (B, CD4 T, CD8 T, monocytes, NK). One study per cell type and gene pair: `eQTLGen_<cellType>_<eGene>_<coeGene>`.

Both have the study type `sceqtl`.

## Run folder

The release read by the platform was copied from the top level of `gs://eqtlgen_data` into a dated run folder on 2026-11-03, before eQTLGen moves to the new credible set schema. Nothing in it was changed.

```sh
gs://eqtlgen_data/prod/20260326T115635/
├── docs/     README.md; 20260326/ holds the README of the bucket as it was
└── output/
    ├── single_cell_eqtl/{260301,260326}/{study,credible_set}
    ├── single_cell_eqtl/{study_index,credible_set}
    ├── coeqtl/{260301,260326}/{study,credible_set}
    ├── coeqtl/{study_index,credible_set}
    └── ccqtl/{study_index,credible_set}
```

The timestamp is the time of the first object of the `260326` release (`coeqtl/260326/credible_set`, 2026-03-26T11:56:35Z).

The output has three generations. Only `260326` is read by `pis`.

| Folder | Written | Content |
|---|---|---|
| `{single_cell_eqtl,coeqtl}/260301/` | 2026-03-02 | credible sets, not filtered |
| `{single_cell_eqtl,coeqtl}/260326/` | 2026-03-26 | `260301` filtered: `pValueExponent <= -3` and `credibleSetlog10BF >= 2 / ln(10)` (0.8686) |
| `{single_cell_eqtl,coeqtl,ccqtl}/{study_index,credible_set}` | 2026-06-25 | a later run, not documented, not read by the platform; `ccqtl` holds 40 studies of type `gwas` |

The data was produced by hand, in a notebook, from the raw files below. No code reproduces `260301` or `260326` exactly.

## Inputs

The SuSiE results provided by eQTLGen stay at the top level of the bucket:

```sh
gs://eqtlgen_data/raw_data/finemap_{B,CD4_T,CD8_T,DC,Mono,NK}/                      # single_cell_eqtl
gs://eqtlgen_data/raw_data/joint_co_eQTL_fine_mapping_{B,CD4_T,CD8_T,Mono,NK}_*/    # coeqtl
gs://eqtlgen_data/raw_data/finemapping_liberal/                                     # ccqtl, not used
```

Each file is a tab-separated table with one row per variant and feature: summary statistics (`z_score`, `p_value`, `beta`, `beta_se`, `n_samples`), the SuSiE credible set of the variant (`SusieRss_CS`) and the log Bayes factors of the 5 SuSiE effects (`lbf_cs_1` to `lbf_cs_5`).

## Release `260326`

| Dataset | Cell type | Credible sets | Studies with credible sets | Study rows | Studies | Biosample |
|---|---|---:|---:|---:|---:|---|
| `single_cell_eqtl` | B | 2,322 | 1,896 | 2,448 | 2,448 | `CL_0000236` |
| | CD4 | 5,833 | 4,143 | 4,977 | 4,977 | `CL_0000084` |
| | CD8 | 2,739 | 2,162 | 2,391 | 2,391 | `CL_0000625` |
| | DC | 433 | 373 | 375 | 375 | `CL_0000451` |
| | Mono | 2,325 | 1,953 | 2,437 | 2,437 | `CL_0000576` |
| | NK | 2,594 | 2,016 | 2,169 | 2,169 | `CL_0000623` |
| | **total** | **16,246** | **12,543** | **14,797** | **14,797** | |
| `coeqtl` | B | 830 | 812 | 1,941 | 897 | `CL_0000236` |
| | CD4 | 8,175 | 7,883 | 12,139 | 8,968 | `CL_0000624` |
| | CD8 | 3,804 | 3,675 | 6,519 | 4,015 | `CL_0000625` |
| | Mono | 1,626 | 1,584 | 2,533 | 1,688 | `CL_0000576` |
| | NK | 756 | 729 | 1,372 | 860 | `CL_0000623` |
| | **total** | **15,191** | **14,683** | **24,504** | **16,428** | |

The credible sets follow the gentropy `StudyLocus` schema of the time (`finemappingMethod` `SuSie`, `locus` with the members). Every credible set has a study.

## Known issues

* The `coeqtl` study index has 24,504 rows for 16,428 study ids: the same study is listed once per input file row, with conflicting `nSamples` ([opentargets/issues#4594](https://github.com/opentargets/issues/issues/4594)).
* The `single_cell_eqtl` effects are of the reference allele, while `variantId` puts the alternative allele last: `beta` and `zScore` have the wrong sign for the effect allele of `variantId`. eQTLGen's `assessed_allele` is the reference allele of `snp_id` (all 9,393 variants of the B cell chromosome 22 file match the GRCh38 reference in gnomAD), and the `260326` leads have the sign of the source `beta` (58 of 58 in that file). The `coeqtl` z-scores were flipped and are correct.
* `SusieRss_CS` numbers the credible sets of a feature, not the SuSiE effects `lbf_cs_1` to `lbf_cs_5` (the credible set 1 can be the effect 3). The notebook took the credible set `k` from `lbf_cs_k` for the `coeqtl`.
* The CD4 T cells are `CL_0000084` (T cell) in `single_cell_eqtl` and `CL_0000624` (CD4-positive, alpha-beta T cell) in `coeqtl`.

## Changelog

### 2026-11-03

* Copied the release from the top level of `gs://eqtlgen_data` into the run folder `gs://eqtlgen_data/prod/20260326T115635/`. `pis` reads the run folder. `raw_data/` stays at the top level, as the input of the next ingestion.
