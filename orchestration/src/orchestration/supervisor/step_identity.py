"""The four names a pipeline step answers to, and how to move between them.

One step is identified four different ways, and code that assumes any two are the
same fails silently rather than loudly:

| Where | Example |
| --- | --- |
| Key in `pis/config.yaml` or `pts/config.yaml` | `disease` |
| Step in `unified_pipeline.yaml`, and the GCP `step` label | `pts_disease` |
| Airflow task group | `pts_disease` |
| Airflow task id of the step's execution task | `pts_disease.run_pts_disease` |

The split is authoritative: `UnifiedPipelineConfig.step_config` documents step names as
`{stage}_{step_name}` and splits on the first underscore, so `pts_evidence_postprocess_impc`
is stage `pts` and config key `evidence_postprocess_impc`.

The qualified task id matters because Airflow prefixes group ids onto their children
(`prefix_group_id` defaults True), so a bare step name never matches a real `task_id`.
That mismatch is invisible: a baseline keyed one way and looked up the other simply never
hits, and stall detection falls back to its ceiling — which is also what a first run looks
like.
"""

from __future__ import annotations

from typing import NamedTuple

RUN_TASK_PREFIX = 'run_'
"""Every stage names its execution task `run_{step}`, inside a group named `{step}`."""


class StepIdentity(NamedTuple):
    """Every name one step answers to.

    Args:
        step: The `unified_pipeline.yaml` step, which is also the GCP `step` label
            and the Airflow task group id.
        stage: The application that runs it, one of `pis`, `pts` or `gentropy`.
        config_key: The key under `steps:` in that stage's own config file.
        run_task_id: The fully-qualified Airflow task id of the execution task.
    """

    step: str
    stage: str
    config_key: str
    run_task_id: str


def identify(step: str) -> StepIdentity:
    """Derive every name for a step from its `unified_pipeline.yaml` name.

    Args:
        step: The step name, in the form `{stage}_{config_key}`.

    Returns:
        The step's identity.

    Raises:
        ValueError: If the name carries no stage prefix, which means it is not a
            `unified_pipeline.yaml` step name and the caller has one of the other
            three spellings.
    """
    stage, _, config_key = step.partition('_')
    if not config_key:
        raise ValueError(
            f'{step!r} has no stage prefix, so it is not a unified_pipeline step name. '
            f'Expected something like "pts_disease".'
        )
    return StepIdentity(
        step=step,
        stage=stage,
        config_key=config_key,
        run_task_id=f'{step}.{RUN_TASK_PREFIX}{step}',
    )


def step_from_task_id(task_id: str) -> str:
    """Recover the step name from any Airflow task id inside its group.

    Args:
        task_id: A task id such as `pts_disease.run_pts_disease`.

    Returns:
        The group component, which is the step name. An unqualified id is returned
        unchanged, since a task outside a group is its own step.
    """
    group, _, _ = task_id.partition('.')
    return group
