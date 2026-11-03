# OTAR2065 microglia single-cell eQTLs

This document was updated on 2026-11-03.

Single-cell eQTLs of human microglia from the OTAR2065 neuroinflammation project, in three conditions (IFN, LPS, untreated), each split into proliferating and non-proliferating cells. There is no publication. The data is read by the Partner Preview Platform (PPP) only.

The eQTL Catalogue team (OTAR2077) mapped the eQTLs with their [qtlmap](https://github.com/eQTL-Catalogue/qtlmap) pipeline on 2025-02-01 (70 phenotype PCs, 6 genotype PCs, 1 Mb cis window, SuSiE with the full output written). Each condition and proliferation state is one eQTL Catalogue dataset (`QTD000759` to `QTD000763`), not in the public eQTL Catalogue.

## Run folder

The release read by `pis` was produced in 2025, outside of this repository, with gentropy 2.x, and written to `gs://eqtl_catalogue_processed_otar_internal_data/otar2077/microglia/gentropy/`. It was copied into a dated run folder of `gs://eqtl_catalogue_data` on 2026-11-03, before the dataset moves to the new credible set schema. Nothing in it was changed.

```sh
gs://eqtl_catalogue_data/otar2065_microglia/prod/20250624T124201/
├── docs/     README.md
└── output/
    ├── study_index/QTD*/    # written 2025-05-15
    ├── study_locus/QTD*/    # written 2025-06-24, gentropy StudyLocus
    └── coloc/QTD*/          # written 2025-04-16, not read by the platform
```

The timestamp is the time of the last write of the release (`study_locus`, 2025-06-24T12:42:01Z).

## Inputs

The qtlmap output stays in the eQTL Catalogue processing bucket:

```sh
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/microglia/susie/QTD*_credible_sets.parquet
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/microglia/susie/QTD*_lbf_variable_merged.parquet
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/microglia/susie_batches/QTD*/{cs,lbf,full}/
gs://eqtl_catalogue_processed_otar_internal_data/otar2077/microglia/sumstats/QTD*/all/
```

The SuSiE files have the same schemas as those of the eQTL Catalogue r8 release.

## Release

| Dataset | Sample group | Condition | Samples | Studies | Credible sets |
|---|---|---|---:|---:|---:|
| `QTD000759` | `microglia_IFN_nonProlif` | IFN | 211 | 2,429 | 3,599 |
| `QTD000760` | `microglia_LPS_Prolif` | LPS | 131 | 287 | 1,227 |
| `QTD000761` | `microglia_LPS_nonProlif` | LPS | 207 | 2,507 | 3,714 |
| `QTD000762` | `microglia_untreated_Prolif` | untreated | 149 | 402 | 1,119 |
| `QTD000763` | `microglia_untreated_nonProlif` | untreated | 208 | 3,150 | 4,741 |
| **total** | | | | **8,775** | **14,400** |

- One study per sample group and gene: `OTAR2065_neuroinflammation_ge_<sample group>_<ENSG>`, project `OTAR2065_neuroinflammation`, study type `sceqtl`, biosample `CL_0000129` (microglial cell).
- 394,699 credible set members, every one with beta, standard error and p-value; the posterior probabilities of each credible set sum to 0.95–1.0. `finemappingMethod` is `SuSie`.

## Known issues

- 3,293 credible sets carry the quality control `Subsignificant p-value` and are still in the release; the threshold used is not recorded.
- `study_index` once held every study twice, under the projects `microglia` and `OTAR2065_neuroinflammation` ([opentargets/issues#3924](https://github.com/opentargets/issues/issues/3924)). The frozen copy has only `OTAR2065_neuroinflammation`, with unique study ids.

## Changelog

### 2026-11-03

* Copied the release from `gs://eqtl_catalogue_processed_otar_internal_data/otar2077/microglia/gentropy/` into the run folder `gs://eqtl_catalogue_data/otar2065_microglia/prod/20250624T124201/`. `pis` reads the run folder. The source folder is not changed.
