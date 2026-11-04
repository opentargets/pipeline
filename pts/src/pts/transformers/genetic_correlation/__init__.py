"""Genetic correlation between GWAS studies, by LD score regression.

Three tasks, run in order by the `genetic_correlation` step:

1. `representatives`: one study per disease and LD ancestry, the best powered of those with a
   usable heritability estimate.
2. `munge`: each representative study's summary statistics joined to its ancestry's LD scores,
   once, as one parquet file per study.
3. `pairs`: `ldsc.genetic_correlation` over every pair of representatives within an ancestry.

The step reads every representative's summary statistics, so it runs ahead of a release as a
staging DAG, not inside it.
"""
