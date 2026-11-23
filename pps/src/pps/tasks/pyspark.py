"""Task to run a pps step function in a spark session."""

import inspect
import os
from typing import Any, Self

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pps.session import Session
from pps.step import resolve_step


class PysparkSpec(Spec):
    """Specification for a pyspark task."""

    step: str
    """The step to run, as `<package>.<function>` in `pps`, e.g. `gwas_catalog.top_hit_ingestion`."""
    settings: dict[str, Any] = {}
    """Settings for the step, validated against the step's settings model."""
    session: dict[str, Any] = {}
    """Keyword arguments for the `pps.session.Session` the step runs in, e.g.
    `write_mode`, `output_partitions` or `start_hail`."""


def session_kwargs(session: dict[str, Any]) -> dict[str, Any]:
    """Keyword arguments for the `Session` of a task, checked against its signature.

    Unless set, the spark master is yarn when running under spark-submit (e.g. as a
    dataproc job), and local otherwise.

    Raises:
        ValueError: If an argument is not accepted by `Session`.
    """
    spark_uri = 'yarn' if 'PYSPARK_GATEWAY_PORT' in os.environ else 'local[*]'
    kwargs = {'spark_uri': spark_uri, 'app_name': 'pps', **session}
    try:
        inspect.signature(Session).bind(**kwargs)
    except TypeError as e:
        raise ValueError(f'invalid session settings {session}: {e}') from e
    return kwargs


class Pyspark(Task):
    """Task to run a pps step function in a spark session.

    The step, its settings and the session settings are resolved and validated when
    the task is built, so a misconfigured step fails before any spark work starts.
    """

    def __init__(self, spec: PysparkSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: PysparkSpec
        self.step, settings_model = resolve_step(spec.step)
        self.settings = settings_model.model_validate(spec.settings)
        self.session_kwargs = session_kwargs(spec.session)

    @report
    def run(self) -> Self:
        logger.info(f'running step {self.spec.step} with settings {self.settings}')
        session = Session(**self.session_kwargs)
        self.step(session, self.settings)
        return self
