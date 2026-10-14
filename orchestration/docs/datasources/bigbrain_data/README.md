# BigBrain cis-eQTL/sQTL

BigBrain is a multi-cohort brain molecular-QTL meta-analysis (43 tissue-cohort pairs,
10,725 samples from 4,656 donors, mmQTL random-effects meta-analysis). It publishes its own
SuSiE fine-mapping on Zenodo, which this DAG ingests as credible sets plus a study index:

- eQTL: [Zenodo record 17226890](https://zenodo.org/records/17226890), one study per gene.
- sQTL: [Zenodo record 17153730](https://zenodo.org/records/17153730), one study per
  Leafcutter intron junction.

Fine-mapping used PolyFun + SuSiE (L=5, residual variance fixed at 1) with in-sample LD
from 3,884 European-ancestry BigBrain donors, so it is ingested directly rather than
re-fine-mapped against a reference LD panel. Re-fine-mapping the summary statistics with
SuSiE-inf and reference-panel LD produced inflated log Bayes factors in genes with several
signals.

```{bash}
gs://genetics-portal-dev-analysis/xg1/bigbrain/raw/{qtl_type}/finemapping/
gs://genetics-portal-dev-analysis/xg1/bigbrain/raw/sqtl/top_assoc_tsv/
gs://genetics-portal-dev-analysis/xg1/bigbrain/published_finemapping/{qtl_type}/credible_set/
gs://genetics-portal-dev-analysis/xg1/bigbrain/published_finemapping/{qtl_type}/study_index/
```

> [!NOTE]
> These paths are under the personal dev prefix `gs://genetics-portal-dev-analysis/xg1/bigbrain`.
> Move them to dedicated buckets before BigBrain becomes a production dataset.

## bigbrain_finemapping_ingestion dag

One `bigbrain_finemapping_ingestion` gentropy step per QTL type, each writing credible sets and
a study index built from the fine-mapped features (so every credible set has a study).

- There is no credible-set QC node, as for eQTL Catalogue: its `purityMinR2` filter would drop
  every multi-variant set because BigBrain does not publish purity. Their sets already passed
  SuSiE's own purity filter (min |r| >= 0.5) on the in-sample LD.
- Credible sets carry PIPs but no log Bayes factors, so they suit the PIP-based colocalisation
  methods (`coloc_pip_ecaviar` in production), not `coloc`.
- Credible sets led by an untested variant (`Z == 0` placeholder rows, about 2%) are dropped.

### Staging inputs

The inputs are staged from Zenodo beforehand, to avoid Zenodo's intermittent 504s:

```{bash}
B=gs://genetics-portal-dev-analysis/xg1/bigbrain/raw
curl -L -o BigBrain_cis_eQTL_finemapping.txt.gz https://zenodo.org/records/17226890/files/BigBrain_cis_eQTL_finemapping.txt.gz
curl -L -o BigBrain_cis_sQTL_finemapping.txt.gz https://zenodo.org/records/17153730/files/BigBrain_cis_sQTL_finemapping.txt.gz
curl -L -o BigBrain_cis_sQTL_EUR_top_assoc.tsv.gz https://zenodo.org/records/17153730/files/BigBrain_cis_sQTL_EUR_top_assoc.tsv.gz
gcloud storage cp BigBrain_cis_eQTL_finemapping.txt.gz $B/eqtl/finemapping/
gcloud storage cp BigBrain_cis_sQTL_finemapping.txt.gz $B/sqtl/finemapping/
gcloud storage cp BigBrain_cis_sQTL_EUR_top_assoc.tsv.gz $B/sqtl/top_assoc_tsv/
```

### Gene assignment

Checked against the 26.03 target index:

| | Studies | Gene not resolved | Credible sets kept after study validation |
| --- | --- | --- | --- |
| eQTL | 16,691 | 363 (retired/novel Ensembl IDs) | 41,228 of 42,082 (98.0%) |
| sQTL | 35,879 | 3,204 (2,989 ambiguous multi-gene, 144 no gene, 71 not in target index) | 62,293 of 68,808 (90.5%) |

About 10% of sQTL junctions are annotated to several overlapping genes. 768 of those resolve
to a single gene (the one whose body contains the junction, or the one protein-coding gene that
does); the rest are left without a gene rather than guessed, and study validation excludes them.

## Changelog

### 2026-10-14

- Added the `bigbrain_finemapping_ingestion` DAG.
