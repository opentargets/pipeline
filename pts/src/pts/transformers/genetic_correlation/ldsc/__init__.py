"""LD score regression in numpy: heritability, genetic covariance and genetic correlation.

A port of the single-annotation path of github.com/bulik/ldsc, checked against it in
`tests/test_ldsc.py`. `rg.genetic_correlation` is the entry point; `regression` and `jackknife`
are the pieces it is built from.
"""

from pts.transformers.genetic_correlation.ldsc.rg import GeneticCorrelation, genetic_correlation

__all__ = ['GeneticCorrelation', 'genetic_correlation']
