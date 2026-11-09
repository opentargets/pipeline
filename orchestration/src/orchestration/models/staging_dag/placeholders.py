"""Placeholders of the staging DAG configs."""

from __future__ import annotations

import re
from typing import Any

PLACEHOLDER_RE = re.compile(r'\{\{([^{}]*)\}\}')
"""A placeholder, `{{name}}`."""
JINJA_TAG_RE = re.compile(r'\{[%#]')
"""The start of a Jinja statement, `{%`, or comment, `{#`."""
RESERVED_PLACEHOLDERS = frozenset({'input', 'output'})
"""Placeholder names reserved for the environment roots, and so invalid as scratchpad keys."""
TIMESTAMP_RE = r'\d{8}T\d{6}'
"""A timestamp, e.g. `20261026T153052`: the name of a run folder, or of a folder of static test inputs."""


def nested_strings(value: Any) -> list[str]:
    """Return every string nested in a value of a parsed YAML document."""
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in nested_strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in nested_strings(v)]
    return []


def check_rendered(value: Any) -> None:
    """Raise if a value of a config contains a placeholder, or a Jinja tag.

    A rendered config contains no placeholder: a placeholder left after the rendering is a
    typo, or a value missing from the environment. The commands of the VM steps are Jinja
    templates, rendered by Airflow at run time, so a Jinja statement (`{%`) or comment (`{#`)
    would break the template or hide a part of it.

    Examples:
        >>> check_rendered({'path': 'gs://bucket/prod/20261026T153052/output', 'conf': "{spark.x: '1'}"})
        >>> check_rendered({'path': '{{output}}/output'})
        Traceback (most recent call last):
            ...
        ValueError: the config contains unrendered placeholders: {{output}}. Read the config with StagingDagConfig, which renders the placeholders of the environment
        >>> check_rendered({'command': 'echo ${#ITEMS[@]}'})
        Traceback (most recent call last):
            ...
        ValueError: the config contains the Jinja tags ['{#']: Airflow renders the commands of the VM steps as Jinja templates

    Raises:
        ValueError: If the value contains a placeholder, or a Jinja tag.
    """  # noqa: E501
    if names := sorted({name for s in nested_strings(value) for name in PLACEHOLDER_RE.findall(s)}):
        raise ValueError(
            'the config contains unrendered placeholders: '
            + ', '.join(f'{{{{{name}}}}}' for name in names)
            + '. Read the config with StagingDagConfig, which renders the placeholders of the environment'
        )
    if tags := sorted({tag for s in nested_strings(value) for tag in JINJA_TAG_RE.findall(s)}):
        raise ValueError(
            f'the config contains the Jinja tags {tags}: '
            'Airflow renders the commands of the VM steps as Jinja templates'
        )
