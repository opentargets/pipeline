"""Tests for release-candidate naming."""

import pytest

from orchestration.utils.release import candidate_names


def test_names_from_a_run_uri():
    names = candidate_names('gs://open-targets-pipeline-runs/do/platform-2612-3')

    assert names.prefix == 'do'
    assert names.run == 'platform-2612-3'
    assert names.product == 'platform'
    assert names.release == '26.12'
    assert names.yymm == '2612'
    assert names.ch_snapshot == 'do-platform-2612-3-ch'
    assert names.os_snapshot == 'do-platform-2612-3-os'
    assert names.ch_disk == 'pos-do-platform-2612-3-ch'
    assert names.os_disk == 'pos-do-platform-2612-3-os'


def test_trailing_slash_is_ignored():
    uri = 'gs://open-targets-pipeline-runs/do/platform-2612-3'
    assert candidate_names(uri + '/').ch_snapshot == candidate_names(uri).ch_snapshot


def test_nested_path_and_run_suffix():
    names = candidate_names('gs://ot-team/dochoa/x/ppp-2612-1-test')

    assert names.prefix == 'x'
    assert names.product == 'ppp'
    assert names.ch_snapshot == 'x-ppp-2612-1-test-ch'


@pytest.mark.parametrize(
    'uri',
    [
        'gs://b/platform-2612-3',
        'gs://b/Do_x/platform-2612-3',
        'gs://b/do/target-2612-3',
        f'gs://b/{"p" * 60}/platform-2612-3',
    ],
)
def test_invalid_uris_raise(uri: str):
    with pytest.raises(ValueError):
        candidate_names(uri)
