"""Evidence-postprocess transformers, one module per datasource.

Grouped here as the number of evidence datasources migrated from PySpark to Polars grows, rather
than left as same-named top-level files directly under `pts.transformers`. Shared validation,
identifier assignment, dating and scoring logic lives in `pts.transformers.evidence.utils`, used by
every module in this package; each module here only holds what's specific to its own datasource.

`pts/config.yaml` addresses a function here with a dotted `transformer:` name, e.g.
`transformer: evidence.gwas_evidence` for `pts.transformers.evidence.gwas_evidence`'s
`gwas_evidence` function -- see `pts.tasks.transform.Transform.load_transformer`.
"""
