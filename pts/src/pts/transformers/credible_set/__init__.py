"""Credible sets: the fine-mapped signals of every study, checked, validated, annotated and published.

A credible set is a set of variants that most likely holds the causal variant of one signal of a
study. Each source stages two datasets, its credible sets and their variants (`source`), and four
stages, in order, turn them into the published `credible_set` and `credible_set_variant`:

1. `variant_qc`, per source: flags each variant (identifiers, coverage, statistics).
2. `set_qc`, per source: flags each set (identifiers, chromosome, MHC, method, study).
3. `validation`, over every source: excludes what the flags and the cross-set QC (`cross_set_qc`)
   exclude, and the ids repeated within a source or kept once across sources; the only stage that
   removes rows.
4. `annotation`, over every source: confidence (`confidence`), TRANS_QTL (`trans_qtl`), REPLICATED
   (`replication`), the effect allele frequency (`allele_frequency`) and the rescaled effect size
   (`rescaling`).

Each stage module opens with its task entry point, the function otter runs; each rule module with
its rule. The flags (`flags`), the fixed vocabularies (`vocabulary`) and the identifiers
(`identifier`) are shared by all of them.
"""
