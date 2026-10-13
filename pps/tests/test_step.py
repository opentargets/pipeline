import pytest

from pps.gwas_catalog.steps import TopHitIngestionSettings, top_hit_ingestion
from pps.step import InvalidStepError, resolve_step


def test_resolve_step_returns_function_and_settings_model():
    function, settings_model = resolve_step('gwas_catalog.top_hit_ingestion')

    assert function is top_hit_ingestion
    assert settings_model is TopHitIngestionSettings


@pytest.mark.parametrize(
    ('reference', 'message'),
    [
        ('top_hit_ingestion', 'expected <package>.<function>'),
        ('missing_package.step', 'module pps.missing_package.steps not found'),
        ('gwas_catalog.missing_step', 'pps.gwas_catalog.steps.missing_step not found'),
        ('gwas_catalog.TopHitIngestionSettings', 'annotated with a pydantic model'),
    ],
)
def test_resolve_step_rejects_invalid_references(reference: str, message: str):
    with pytest.raises(InvalidStepError, match=message):
        resolve_step(reference)
