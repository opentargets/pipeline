import subprocess
import sys

from pts.transformers.clinical_mira.llm_schema import patch_schema, result_key
from pts.transformers.clinical_mira.schemas import RoleDrug, TrialExtractionV2, TrialLink


def test_claude_batches_import_no_openai() -> None:
    code = "import sys; import pts.transformers.clinical_mira.moa.claude; print('openai' in sys.modules)"
    out = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


def test_patch_schema_closes_every_object() -> None:
    schema = patch_schema(TrialExtractionV2.model_json_schema(by_alias=True))
    objects = [schema, *schema['$defs'].values()]
    for node in objects:
        if 'properties' in node:
            assert node['additionalProperties'] is False
            assert node['required'] == list(node['properties'])


def test_old_names_still_importable() -> None:
    from pts.transformers.clinical_mira.llm_workflow import _patch_schema
    from pts.transformers.clinical_mira.moa.llm import result_key as old_result_key

    schema = TrialExtractionV2.model_json_schema(by_alias=True)
    assert _patch_schema(schema) == patch_schema(schema)
    assert old_result_key is result_key


def test_trial_link_carries_synonyms() -> None:
    for model in (TrialLink, RoleDrug):
        assert 'synonyms' in model.model_json_schema()['required']
