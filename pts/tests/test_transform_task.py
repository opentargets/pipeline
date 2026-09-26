"""Tests for pts.tasks.transform.Transform.load_transformer."""

import pytest

from pts.tasks.transform import Transform


class TestLoadTransformer:
    def test_loads_a_flat_transformer_name(self) -> None:
        transformer = Transform.load_transformer('disease')

        assert callable(transformer)
        assert getattr(transformer, '__name__', None) == 'disease'

    def test_loads_a_dotted_transformer_name_from_a_subpackage(self) -> None:
        transformer = Transform.load_transformer('evidence.gwas_evidence')

        assert callable(transformer)
        assert getattr(transformer, '__name__', None) == 'gwas_evidence'

    def test_raises_for_an_unknown_transformer(self) -> None:
        with pytest.raises(ModuleNotFoundError):
            Transform.load_transformer('does_not_exist')
