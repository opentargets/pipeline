"""Quality-control flag vocabulary for evidence validation.

Values must match the PySpark `EvidenceFlags` enum (`pts.pyspark.evidence_utils.evidence`)
exactly: `release_metrics.py` filters `qualityControls` on these literal strings across every
evidence datasource, PySpark-produced or Polars-produced alike.
"""

INVALID_DISEASE = 'No valid disease'
INVALID_TARGET = 'No valid target'
DUPLICATED = 'Duplicated'
NO_VALID_SCORE = 'No valid score'
INVALID_BIOTYPE = 'Invalid biotype'
