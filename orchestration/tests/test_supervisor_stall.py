"""Tests for stall detection constants that are coupled to things outside Python."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from orchestration.supervisor.stall import _RUN_STALL_WAKEUP_THRESHOLD

REPO = Path(__file__).resolve().parents[2]
STARTUP = REPO / 'orchestration/deployment/startup_machine.sh'

CRON_INTERVAL = re.compile(r"CRON_LINE='\*/(\d+) \* \* \* \* orchestration ")


def observer_cron_interval_minutes() -> int:
    """Minutes between observer wakeups, read from the shell script that installs the cron.

    Read rather than restated: a copy of the number in this file would agree with itself
    forever while the deployed crontab drifted away from it, which is the exact failure
    this module exists to prevent.

    Returns:
        The `*/N` interval from `startup_machine.sh`'s `CRON_LINE`.
    """
    match = CRON_INTERVAL.search(STARTUP.read_text())
    if match is None:
        pytest.fail(
            f'no CRON_LINE matching {CRON_INTERVAL.pattern!r} in {STARTUP}. The observer cron '
            'was renamed, reformatted or removed. That is not necessarily wrong, but it means '
            'nothing is checking the wakeup threshold against the real cadence any more — fix '
            'this pattern rather than deleting the test.'
        )
    return int(match.group(1))


class TestTheNoProgressThresholdIsCoupledToTheCron:
    """`_RUN_STALL_WAKEUP_THRESHOLD` counts wakeups, so its meaning depends on the cron.

    It is deliberately a count and not a duration — see its docstring: a cron that was
    itself down for a while must not manufacture a false alarm the moment it comes back.
    The cost of that choice is that the constant silently rescales whenever the cron
    interval changes, in a file a Python developer has no reason to open. These tests are
    what makes that impossible to do unnoticed.
    """

    def test_the_no_progress_threshold_still_means_one_hour(self) -> None:
        minutes = observer_cron_interval_minutes() * _RUN_STALL_WAKEUP_THRESHOLD
        assert minutes == 60, (
            f'{_RUN_STALL_WAKEUP_THRESHOLD} wakeups at a '
            f'{observer_cron_interval_minutes()}-minute cadence is {minutes} minutes, not the '
            'one hour the threshold is documented to mean. Either the cron interval in '
            'startup_machine.sh or _RUN_STALL_WAKEUP_THRESHOLD was changed without the other. '
            'Both encode one decision; change them together, or change what the docstring '
            'claims and update this test deliberately.'
        )

    def test_the_cron_interval_is_actually_read_from_the_script(self) -> None:
        """Guards the regex itself, which is the only thing holding the coupling up.

        A pattern that silently stopped matching would make the assertion above fail loudly
        rather than pass vacuously — `pytest.fail` in the helper, not a default — but this
        pins the value it reads so a pattern that matched the *wrong* number is caught too.
        """
        assert observer_cron_interval_minutes() == 5
