"""Genetic correlation between GWAS studies, by LD score regression.

Three tasks, run in order by the `genetic_correlation` staging DAG:

1. `representatives.representatives`: one study per disease and LD ancestry, the best powered of
   those with a usable heritability estimate. Polars, on the driver.
2. `pts.pyspark.genetic_correlation_munge`: each representative's summary statistics joined to its
   ancestry's LD scores, once, as one parquet file per study. One Spark task per study.
3. `pts.pyspark.genetic_correlation_pairs`: `ldsc.genetic_correlation` over every pair of
   representatives within an ancestry. One Spark task per chunk of pairs.

The work of steps 2 and 3 is `munge.munge_one` and `pairs.run_chunk`; Spark only distributes it.
"""
