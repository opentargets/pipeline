"""Tests for loading transformers by name."""

import pytest

from pts.tasks.transform import Transform
from pts.transformers.evidence.encore import encore
from pts.transformers.reactome import reactome


def test_loads_a_plain_transformer_name() -> None:
    assert Transform.load_transformer('reactome') is reactome


def test_loads_a_dotted_transformer_name_from_a_subpackage() -> None:
    assert Transform.load_transformer('evidence.encore') is encore


def test_raises_for_an_unknown_transformer() -> None:
    with pytest.raises(ModuleNotFoundError):
        Transform.load_transformer('does_not_exist')
