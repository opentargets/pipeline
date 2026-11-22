"""Models of the views pts computes from the outputs, one module per view, written to `view/<dataset_name>`.

Views are not release outputs: they are published in some contexts and not others, so
`discover_datasets()` leaves them out and `discover_models()` adds them. A column copied from an output
is derived from that output's model (`ot_datamodels.derive`); only computed columns are declared here.
"""
