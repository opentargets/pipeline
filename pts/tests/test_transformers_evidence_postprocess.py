"""Tests for the `evidence_postprocess` transformer entry point.

The step is thin by design, so there is little here: reading is covered in `test_evidence_read.py`,
the recipe in `test_evidence_postprocess.py`, the `Evidence` chain and LUT builders in
`test_evidence_polars.py`, and the shared writer in `test_dataset.py`. What only this module owns
is the registry lookup, which is what binds a `datasource_id` to its expressions.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pts.transformers.evidence_postprocess import evidence_postprocess


def test_evidence_postprocess_raises_clearly_for_an_unregistered_datasource(tmp_path: Path) -> None:
    """The registry lookup happens before any LUT is built or file is read, so this fails fast.

    Asserts the CUSTOM message's own wording ('no score/direction expressions registered'), not
    just the datasource id: a bare `EXPRESSIONS[id]` KeyError also carries the id in its message
    (`KeyError: 'not_a_real_datasource'`), so matching on the id alone would pass even if the
    `try/except` that builds the clearer message were deleted.
    """
    settings = {'datasource_id': 'not_a_real_datasource', 'evidence_format': 'parquet', 'unique_fields': []}
    missing = str(tmp_path / 'does_not_exist')
    source = {
        'evidence_path': missing,
        'target_path': missing,
        'disease_path': missing,
        'publication_date_lut': missing,
    }

    with pytest.raises(KeyError, match='no score/direction expressions registered for datasource'):
        evidence_postprocess(source, {}, settings, None)
