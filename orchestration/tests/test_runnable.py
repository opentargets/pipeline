"""Tests for RunnableSpec script delivery against real asset scripts."""

from importlib.resources import files

import pytest

from orchestration.models.batch.runnable import RunnableSpec

ANNOTATE_TRANSCRIPTS = files('orchestration.assets').joinpath('annotate_transcripts.sh').read_text()
ANNOTATE_VARIANTS = files('orchestration.assets').joinpath('annotate_variants.sh').read_text()
HARMONISE = files('orchestration.assets').joinpath('harmonise.sh').read_text()


@pytest.mark.parametrize(
    ('script_file', 'expected_content'),
    [
        pytest.param('annotate_transcripts.sh', ANNOTATE_TRANSCRIPTS, id='annotate_transcripts'),
        pytest.param('annotate_variants.sh', ANNOTATE_VARIANTS, id='annotate_variants'),
        pytest.param('harmonise.sh', HARMONISE, id='harmonise'),
    ],
)
def test_script_file_is_delivered_verbatim(script_file: str, expected_content: str) -> None:
    """A script_file must reach the container as `bash -c <verbatim script>` with no rewriting."""
    spec = RunnableSpec(image_uri='gcr.io/project/image:latest', script_file=script_file)
    assert spec.commands == ['-c', expected_content]


def test_script_variables_are_templated_before_delivery() -> None:
    """`${var}` sentinels are substituted, and the rest of the script is left untouched."""
    spec = RunnableSpec(
        image_uri='gcr.io/project/image:latest',
        script_file='harmonise.sh',
        script_variables={'RAW': 'gs://bucket/raw.tsv.gz'},
    )
    _, script = spec.commands
    assert 'gs://bucket/raw.tsv.gz' in script
    assert '${RAW}' not in script
    # Untemplated sentinels are preserved verbatim for the container to resolve at runtime.
    assert '${HARMONISED}' in script


def test_inline_commands_are_used_directly() -> None:
    spec = RunnableSpec(image_uri='gcr.io/project/image:latest', inline_commands=['echo', 'hello'])
    assert spec.commands == ['echo', 'hello']
