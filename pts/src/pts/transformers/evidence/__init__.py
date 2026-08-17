"""Evidence generation and post-processing.

Home for everything that produces or processes Open Targets Platform evidence in polars:

* `core` -- the `Evidence` chain, one validation or enrichment per method.
* `expressions` -- score and direction-of-effect expressions, keyed by datasource.
* `postprocess` -- the parametrised recipe that runs the chain in order.

Nothing here reads or writes. Storage belongs to whoever drives the recipe.

The `evidence_postprocess_*` step lives one level up, in `pts.transformers.evidence_postprocess`,
because otter resolves a transformer by importing `pts.transformers.<name>` and taking the
attribute of the same name -- a step cannot live inside a subpackage without changing that loader.

This package is also where per-datasource evidence generation belongs: one module each, running the
shared recipe on the evidence it produces. `postprocess` takes expressions as a parameter and never
looks them up, so a module can supply its own rather than registering them centrally.
"""
