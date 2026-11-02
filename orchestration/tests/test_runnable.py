"""Tests for RunnableSpec script parsing methods against real asset scripts."""

from importlib.resources import files

import pytest

from orchestration.models.batch.runnable import RunnableSpec

SUSIE_FINEMAP = files('orchestration.assets').joinpath('susie_finemap.sh').read_text()


@pytest.mark.parametrize(
    ('script', 'expected_line_count'),
    [
        pytest.param(SUSIE_FINEMAP, 31, id='susie_finemap'),
    ],
)
def test_remove_comments_line_count(script: str, expected_line_count: int) -> None:
    assert len(RunnableSpec._remove_comments(script)) == expected_line_count


@pytest.mark.parametrize(
    ('script', 'expected_count'),
    [
        pytest.param(SUSIE_FINEMAP, 4, id='susie_finemap'),
    ],
)
def test_join_continuation_lines_count(script: str, expected_count: int) -> None:
    lines = RunnableSpec._remove_comments(script)
    assert len(RunnableSpec._join_continuation_lines(lines)) == expected_count


@pytest.mark.parametrize(
    ('script', 'expected_count'),
    [
        pytest.param(SUSIE_FINEMAP, 4, id='susie_finemap'),
    ],
)
def test_split_commands_count(script: str, expected_count: int) -> None:
    lines = RunnableSpec._remove_comments(script)
    joined = RunnableSpec._join_continuation_lines(lines)
    assert len(RunnableSpec._split_commands(joined)) == expected_count


@pytest.mark.parametrize(
    ('script', 'expected_count'),
    [
        pytest.param(SUSIE_FINEMAP, 4, id='susie_finemap'),
    ],
)
def test_parse_script_file_command_count(script: str, expected_count: int) -> None:
    result = RunnableSpec._parse_script_file(script)
    assert len(result.split(' && ')) == expected_count


def test_parse_script_file_preserves_or_guards() -> None:
    """Parser must not split || guard clauses into separate && commands."""
    script = '[[ -n "${X}" ]] || { echo "X is not set"; exit 1; }'
    result = RunnableSpec._parse_script_file(script)
    assert '|| {' in result, '|| guard clauses must be preserved, not split into &&'
