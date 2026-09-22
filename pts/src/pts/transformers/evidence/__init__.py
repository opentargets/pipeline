"""Reusable Polars evidence-validation building blocks.

Polars port of `pts.pyspark.evidence_utils` (`evidence.py` + `validation_lut.py`), covering the
parts every evidence-postprocess step needs: disease/target resolution, uniqueness, dating and
scoring. Datasource-specific evidence *generation* is not here -- see e.g.
`pts.transformers.gwas_evidence` for the first consumer.
"""

from pts.transformers.evidence.dating import resolve_evidence_date, resolve_publication_date
from pts.transformers.evidence.identifiers import assign_evidence_identifier, validate_uniqueness
from pts.transformers.evidence.luts import build_disease_lut, build_publication_lut, build_target_lut
from pts.transformers.evidence.scoring import calculate_evidence_score
from pts.transformers.evidence.validation import validate_diseases, validate_target

__all__ = [
    'assign_evidence_identifier',
    'build_disease_lut',
    'build_publication_lut',
    'build_target_lut',
    'calculate_evidence_score',
    'resolve_evidence_date',
    'resolve_publication_date',
    'validate_diseases',
    'validate_target',
    'validate_uniqueness',
]
