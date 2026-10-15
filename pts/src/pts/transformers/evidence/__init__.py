"""Evidence generation and post-processing.

Home for everything that produces or processes Open Targets Platform evidence in polars:

* `utils.evidence` -- the `Evidence` chain, one validation or enrichment per method.
* `utils.expressions` -- score helpers shared by the datasource modules.
* `utils.datasource` -- the parametrised recipe that runs the chain in order.
* One module per datasource, `evidence.<name>`, which is also the transformer a step names in
  `config.yaml`.
"""
