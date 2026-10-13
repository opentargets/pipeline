"""Task to run a pps step function in a spark session."""

from typing import Any, Self

from loguru import logger
from otter.task.model import Spec, Task, TaskContext
from otter.task.task_reporter import report

from pps.session import Session, SessionSettings
from pps.step import resolve_step


class PysparkSpec(Spec):
    """Specification for a pyspark task."""

    step: str
    """The step to run, as `<package>.<function>` in `pps`, e.g. `gwas_catalog.top_hit_ingestion`."""
    settings: dict[str, Any] = {}
    """Settings for the step, validated against the step's settings model."""
    session: SessionSettings = SessionSettings()
    """Settings for the spark session the step runs in."""


class Pyspark(Task):
    """Task to run a pps step function in a spark session.

    The step and its settings are resolved and validated when the task is built,
    so a misconfigured step fails before any spark work starts.
    """

    def __init__(self, spec: PysparkSpec, context: TaskContext) -> None:
        super().__init__(spec, context)
        self.spec: PysparkSpec
        self.step, settings_model = resolve_step(spec.step)
        self.settings = settings_model.model_validate(spec.settings)

    @report
    def run(self) -> Self:
        logger.info(f'running step {self.spec.step} with settings {self.settings}')
        session = Session(self.spec.session)
        self.step(session, self.settings)
        return self
