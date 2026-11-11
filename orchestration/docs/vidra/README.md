# VIDRA in the unified pipeline

VIDRA (variant-informed dose-response analysis) regresses disease risk on protein function across
the variants of a gene-disease pair, and reports the posterior of that slope. It was a standalone
three-step pipeline on Dataproc Serverless and a VEP annotation VM
([TrynkaLab/VIDRA](https://github.com/TrynkaLab/VIDRA), branch `vidra_2`); the unified pipeline
runs it as pts steps on the platform's own datasets.

## Steps

| Step | Runs on | Reads | Writes |
| --- | --- | --- | --- |
| `pis_vidra` | pis VM | the AstraZeneca PheWAS Portal binary ExWAS export (`gs://ot-team/dconsidine/vidra/`, to move next to the portal's burden files) | `input/vidra/` |
| `pts_vidra_az_exwas` | pts VM (`vidra.az_exwas`) | the export | `intermediate/vidra/az_exwas`: allelic-model associations at p < 5e-8, with platform variant ids |
| `pts_variant` | as today | `additional_variants: [intermediate/vidra/az_exwas]` | the variant index also annotates VIDRA's rare variants, 30% of which are in neither ClinVar nor a credible set |
| `pts_vidra_input` | `pts_vidra` Dataproc cluster (`pyspark vidra_input`) | `output/{colocalisation,credible_set,credible_set_variant,study,variant,target,disease,biosample,evidence_eva}`, the ExWAS rows, the raw AZ collapsing analyses (`input/evidence/gene_burden/astrazeneca/`), the OnToma label lookup | `intermediate/vidra/analysis_ready`, one row per variant, gene, disease and source, partitioned by gene |
| `pts_vidra_bayes` | `pts_vidra` cluster (`pyspark vidra_bayes`) | `intermediate/vidra/analysis_ready` | `output/vidra`: one row per gene, disease and parameter (see `pts.schemas.vidra`) |

The analysis-ready table keeps VIDRA's original column names (`variant`, `as_gene`, `as_disease`,
`GsourceLab`, `GqtlLab`, `yc`, `ycse`, `xc`, `xcse`, `bO`, `bOse`, `as_clinicalSignificance`) and adds the
protein-function annotations and provenance columns; the published dataset uses platform names.

### Sources of a row (`GsourceLab`)

- **0, common variants**: a GWAS credible set colocalising (COLOC H4 > 0.7) with an eQTL, single-cell eQTL
  or pQTL credible set of a protein-coding gene in blood, blood plasma or a haematopoietic cell type
  (`biosample` ancestors). The GWAS lead is the variant; `yc` is its rescaled effect in the GWAS
  (`credible_set_variant.rescaledAbsBeta`, signed), `xc` the QTL lead's reported effect in the QTL study.
  The two leads are in LD but need not be the same variant: the platform does not carry the GWAS lead's
  effect inside the QTL study, which the standalone pipeline used.
- **1, rare variants**: AstraZeneca ExWAS associations, `yc = log(OR)`, phenotypes mapped to EFO with the
  OnToma label lookup (86% of the portal's phenotype strings map; the standalone pipeline used a private
  curation file).
- **2, ClinVar**: EVA evidence with its clinical significance scored on `[0, 1]` (the new
  "conflicting classifications of pathogenicity" wording scores as uncertain). `directionOnTarget` and
  `directionOnTrait` are carried through for a future gain-of-function flag.
- **3, coding GWAS**: genome-wide significant GWAS leads with a coding consequence on the gene's
  canonical transcript, not in any colocalising set.

Every row gets the gene-disease burden test of the AstraZeneca collapsing analyses (`ptv`, `ptv5pcnt`,
`ptvraredmg` models; `bO`, `bOse`), read from the raw portal files rather than the p-value-filtered
gene burden evidence.

### Annotations

The VEP annotation VM is gone: protein-function scores come from the variant index per variant and gene.
`as_gpnstar = 1 - sigmoid(a + b * llr)` maps the GPN-Star calibrated score to a benign probability; `a`
and `b` are fitted on ClinVar missense variants by `pts/tools/vidra_gpn_star_calibration.py` and live in
the `vidra_input` settings with the run they were fitted on. `as_alphamissense = 1 - pathogenicity`.
The consequence on the canonical transcript classifies each row for imputation in the Bayesian step:
loss of function (including LOFTEE high confidence) is damaging, missense without a score takes the
gene's missense mean, synonymous and non-coding changes are benign.

## CmdStan on the cluster

The Stan models (`pts/src/pts/stan/vidra/`) are not compiled on the cluster. `pts/tools/build_vidra_stan.sh`
builds CmdStan and the two executables at `/opt/vidra` in a Debian 13 container and writes a tarball
whose name carries the CmdStan version and the hash of the `.stan` files. Upload it and point
`vidra_stan_uri` in `unified_pipeline.yaml` at it; the `pts_vidra` cluster's second init action
(`install_vidra_stan.sh`) extracts it on every node, and the pts init action installs cmdstanpy through
the cluster's `PTS_EXTRAS: vidra` metadata. A change to a `.stan` file needs a new tarball.

## Validation

The standalone pipeline's validation set (`scripts/validation/validation_set.csv` in the VIDRA repo, 2,201
gene-disease pairs) is the comparison set for a port run: `vidra_bayes` takes a `genes` setting (a text
file of gene ids) to restrict a run. Differences to expect against the published VIDRA run: rows lost
where a GWAS lead has no rescaled effect (33% of PICS top hits), the QTL-lead `xc` semantics, and the new
annotation set.
