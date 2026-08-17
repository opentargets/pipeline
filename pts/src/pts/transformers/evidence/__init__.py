"""Evidence generation and post-processing.

Home for everything that produces or processes Open Targets Platform evidence in polars.

Today it holds the shared machinery: the `Evidence` chain (`core`), the per-datasource score and
direction-of-effect registry (`expressions`), and the parametrised post-processing recipe
(`postprocess`). The `evidence_postprocess_*` step that drives all of it lives one level up, in
`pts.transformers.evidence_postprocess`, because otter resolves a transformer by importing
`pts.transformers.<name>` and taking the attribute of the same name -- a step cannot live inside a
subpackage without changing that loader.

Nothing here reads or writes. Storage belongs to whoever drives the recipe: the step today, a
per-datasource module later.

The direction of travel is one module PER DATASOURCE here, each owning its own evidence generation
-- in polars where the port allows, otherwise generated in spark and read back as parquet -- and
each running the shared recipe on the result. The `expressions` registry dissolves into those
modules as they land, which is why `postprocess` takes expressions as a parameter and never looks
them up.
"""
