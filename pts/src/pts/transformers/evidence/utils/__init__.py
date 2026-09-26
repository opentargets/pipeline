"""Reusable Polars evidence-validation building blocks.

Polars port of `pts.pyspark.evidence_utils` (`evidence.py` + `validation_lut.py`), covering the
parts every evidence-postprocess step needs: disease/target resolution, uniqueness, dating and
scoring. Datasource-specific evidence *generation* is not here -- see the sibling modules in
`pts.transformers.evidence` (e.g. `pts.transformers.evidence.gwas_evidence`, the first consumer).
"""

from pts.transformers.evidence.utils.dating import resolve_evidence_date, resolve_publication_date
from pts.transformers.evidence.utils.identifiers import assign_evidence_identifier, validate_uniqueness
from pts.transformers.evidence.utils.luts import build_disease_lut, build_publication_lut, build_target_lut
from pts.transformers.evidence.utils.scoring import calculate_evidence_score
from pts.transformers.evidence.utils.validation import validate_datasource, validate_diseases, validate_target

__all__ = [
    'assign_evidence_identifier',
    'build_disease_lut',
    'build_publication_lut',
    'build_target_lut',
    'calculate_evidence_score',
    'resolve_evidence_date',
    'resolve_publication_date',
    'validate_datasource',
    'validate_diseases',
    'validate_target',
    'validate_uniqueness',
]
