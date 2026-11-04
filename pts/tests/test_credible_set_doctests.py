"""The examples in the credible set docstrings run.

PTS does not collect doctests (`testpaths = ["tests"]`), so the modules with examples are run here.
"""

import doctest
from types import ModuleType

import pytest

from pts.transformers.credible_set import (
    annotation,
    replication,
    rescaling,
    set_qc,
    trans_qtl,
    validation,
    variant_id,
    variant_qc,
)


@pytest.mark.parametrize(
    'module',
    [variant_id, variant_qc, set_qc, validation, trans_qtl, replication, rescaling, annotation],
    ids=lambda module: module.__name__,
)
def test_docstring_examples(module: ModuleType) -> None:
    flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL
    result = doctest.testmod(module, optionflags=flags)
    assert result.attempted > 0
    assert result.failed == 0
