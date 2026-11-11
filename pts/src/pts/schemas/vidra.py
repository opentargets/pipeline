"""Schema of the VIDRA dataset: the dose-response posterior summaries of every gene-disease pair."""

from __future__ import annotations

import polars as pl

vidra_schema = pl.Schema({
    'targetId': pl.String,
    'diseaseId': pl.String,
    'parameter': pl.String,
    'model': pl.String,
    'variantCount': pl.Int32,
    'sources': pl.List(pl.Int32),
    'qtlTypes': pl.List(pl.Int32),
    'hasBurden': pl.Boolean,
    'error': pl.String,
    'mean': pl.Float64,
    'median': pl.Float64,
    'percentile1': pl.Float64,
    'percentile2_5': pl.Float64,
    'percentile5': pl.Float64,
    'percentile10': pl.Float64,
    'percentile25': pl.Float64,
    'percentile40': pl.Float64,
    'percentile50': pl.Float64,
    'percentile60': pl.Float64,
    'percentile75': pl.Float64,
    'percentile90': pl.Float64,
    'percentile95': pl.Float64,
    'percentile97_5': pl.Float64,
    'percentile99': pl.Float64,
    'probabilityPositive': pl.Float64,
    'probabilityNegative': pl.Float64,
})
"""One row per gene, disease and fitted parameter.

`parameter` is `slope` (the shared dose-response slope), `intercept`, one of the per-source slopes
`slope_eqtl`, `slope_pqtl`, `slope_coding_gwas`, `slope_az_rare`, `slope_clinvar`, or `meta_slope`,
the pool of the per-source slopes of the sources present. `model` names the fit:
`multiple_variant_fullrank` / `multiple_variant_meanfield` (several variants, ADVI algorithm),
`single_variant`, `single_variant_unidentified` (one variant that cannot identify a slope: no
posterior columns) or `error_report` (the fit failed; `error` holds the message). `sources` are the
evidence sources of the pair (0 common QTL, 1 AstraZeneca rare, 2 ClinVar, 3 coding GWAS) and
`qtlTypes` their QTL branches (0 eQTL, 1 pQTL, 2 none). `probabilityPositive` / `probabilityNegative`
are the posterior probabilities that the parameter is above / below zero.
"""
