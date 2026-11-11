data {
// Hyperparameters
  real<lower=0> h1;
// Observations
  int<lower=0> N; // Number of observations
  // int<lower=0> K; // Number of total G1 (i.e. number of sources)
// Groups
  real numG1; // Number of sources. at the moment AZ, clinvar and GWAS-QTL
  real numG2; // Subgroup indicators qtl type
// Intercept - Burden test (gene-level effect of full loss of function)
  real bO;
  real<lower=0> bOse;
  int<lower=0, upper=1> has_burden;
// // Measure X - protein function
// QTLs
  real xc;
  real xcse;
// Protein predictions (0 = damaging, 1 = benign), as in VIDRA.stan
  real as_gpnstar;
  real as_alphamissense;
// // Measure Y - disease risk
// This are coming from GWAS (including AZ rare-variants one)
  real yOR; // Response variable
  real yORse; // Response variable
// Phenotype predictions
  real as_clinicalSignificance;
}
parameters {
  // Vectors with common variants effects
  real xcest;
  real yORest;
  real slope; // Slope for protein function
  real intercept_random; // Intercept (informed by burden test bO)
  real protein_prior;
  real disease_prior;
}
model {
// Protein (same scores and sds as VIDRA.stan)
protein_prior ~ normal( as_gpnstar, .28);
protein_prior ~ normal( as_alphamissense, .3);
// Disease
disease_prior ~ normal( as_clinicalSignificance, .2);
slope ~ normal( 0, 5 );
// Weak priors for latent effect-size parameters — ensures proper posterior
// even when not constrained by the likelihood branches below
xcest ~ normal(0, 0.2);
yORest ~ normal(0, 1.0);
// Intercept prior — matches VIDRA.stan
intercept_random ~ normal(0, 10);

// Measurement model for the observed effect — applied for every source that
// has a measured yOR (QTL and AZ). ClinVar yOR is a fill value (0.0), not a
// measurement, and ClinVar pairs are not fitted by this model (see below).
yOR ~ normal( yORest, yORse);

// Burden test informs the intercept, as in VIDRA.stan
if (has_burden == 1) {
  bO ~ normal(intercept_random, bOse);
}

if (numG1 == 0) { // QTL common variants (eQTL or pQTL)
  xc ~ normal( xcest, xcse);
  // Wald ratio with delta-method (second-order) SE, which includes the QTL
  // uncertainty xcse. The legacy SD abs(yORse/xcest) ignored xcse and divided by
  // a parameter, giving unreliable fits for weakly measured QTLs.
  // Guard against division by zero — xc is DATA (0.0 when QTL effect missing)
  if (abs(xc) > 1e-4)
    slope ~ normal(yOR / xc, sqrt(square(yORse / xc) + square(yOR * xcse / square(xc))));
  }
else
if (numG1 == 1) { // AZ PheWAS rare variants
  // Same line as the VIDRA.stan AZ branch: the burden intercept is the effect at
  // full loss of function (protein_prior = 0), the variant is a second point on
  // the line. Only fitted when has_burden == 1 — without the burden test the
  // intercept and slope are not separately identified from a single variant.
  yORest ~ normal(intercept_random + slope * protein_prior, 0.1);
}
// ClinVar (numG1 == 2) and coding GWAS (numG1 == 3) are not fitted here:
// a single ClinVar variant has no measured effect size, so its slope is not
// identified (run_bayesian_analysis.py emits a 'single_variant_unidentified'
// row instead), and single-variant coding GWAS pairs are filtered out upstream.
}
