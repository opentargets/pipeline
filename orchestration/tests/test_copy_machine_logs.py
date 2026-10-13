"""Tests for mirroring a machine's logs into the Airflow task log.

Copying logs is presentation: it must not decide the outcome of a step. That makes the
`except` a deliberate behaviour rather than defensive noise, so it is pinned here --
along with the fact that a working copy still reaches the task log.
"""

from unittest.mock import MagicMock, patch

import pytest
from google.api_core.exceptions import ResourceExhausted
from google.cloud.logging_v2.types import LogEntry

from orchestration.operators.gce import ComputeEngineRunContainerizedWorkloadSensor


def _sensor() -> ComputeEngineRunContainerizedWorkloadSensor:
    return ComputeEngineRunContainerizedWorkloadSensor(
        task_id='a_step',
        instance_name='a-machine',
        container_image='an-image',
    )


def _entry(message: str) -> LogEntry:
    entry = LogEntry(log_name='projects/a-project/logs/l')
    entry.json_payload = {'message': message}  # ty:ignore[invalid-assignment]
    return entry


def _with_entries(sensor, entries):
    client = MagicMock()
    client.list_entries.return_value = iter(entries)
    hook = MagicMock()
    hook.get_conn.return_value = client
    return patch.object(type(sensor), 'logging_hook', property(lambda _: hook))


def test_the_messages_reach_the_task_log() -> None:
    sensor = _sensor()

    with _with_entries(sensor, [_entry('first'), _entry('second')]), patch.object(type(sensor), 'log') as log:
        sensor.copy_machine_logs()

    assert [call.args[0] for call in log.info.call_args_list] == ['first', 'second']


def test_an_entry_without_a_message_does_not_break_the_copy() -> None:
    """An entry with no jsonPayload at all reads as None, not as an empty mapping."""
    sensor = _sensor()

    with _with_entries(sensor, [LogEntry(log_name='l'), _entry('second')]), patch.object(type(sensor), 'log') as log:
        sensor.copy_machine_logs()

    assert [call.args[0] for call in log.info.call_args_list] == ['Empty log message', 'second']


@pytest.mark.parametrize(
    'error',
    [ResourceExhausted('quota exceeded'), RuntimeError('a bug in the paging code')],
    ids=['quota', 'bug'],
)
def test_a_failure_to_copy_the_logs_does_not_fail_the_step(error: Exception) -> None:
    """The work has already succeeded by this point; losing its logs must not undo that.

    This swallows real bugs as well as quota errors, which is why the paging code above
    carries its own tests rather than relying on a failure surfacing here.
    """
    sensor = _sensor()
    client = MagicMock()
    client.list_entries.side_effect = error
    hook = MagicMock()
    hook.get_conn.return_value = client

    with (
        patch.object(type(sensor), 'logging_hook', property(lambda _: hook)),
        patch.object(type(sensor), 'log') as log,
    ):
        sensor.copy_machine_logs()  # must not raise

    assert log.warning.call_count == 1
