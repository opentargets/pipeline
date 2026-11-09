"""The examples in the credible set docstrings run.

PTS does not collect doctests (`testpaths = ["tests"]`), so every module of the package is run here,
found rather than listed, so that a new or renamed module is not missed.
"""

import doctest
import importlib
import pkgutil

import pytest

from pts.transformers import credible_set

MODULE_NAMES = sorted(module.name for module in pkgutil.iter_modules(credible_set.__path__))


@pytest.mark.parametrize('module_name', MODULE_NAMES)
def test_docstring_examples(module_name: str) -> None:
    module = importlib.import_module(f'{credible_set.__name__}.{module_name}')
    flags = doctest.NORMALIZE_WHITESPACE | doctest.ELLIPSIS | doctest.IGNORE_EXCEPTION_DETAIL
    assert doctest.testmod(module, optionflags=flags).failed == 0
