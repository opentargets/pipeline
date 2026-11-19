#!/usr/bin/env python3
"""Run one pts task with explicit paths, as a Dataproc PySpark job of a staging DAG.

Usage: dataproc_pts_task_run.py <transformer|pyspark> <name> <spec>

`name` is the module of the task in `pts.transformers` or `pts.pyspark`, as in `config.yaml`.
`spec` is a JSON object with the task's `source`, `destination`, and optional `settings` and
`properties` (Spark properties, pyspark tasks only). Unlike `dataproc_pts_run.py`, no config file
is read: the staging DAG gives every path.
"""

import json
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

from pts.tasks.pyspark import Pyspark
from pts.tasks.transform import Transform


def main() -> None:
    kind, name, spec = sys.argv[1], sys.argv[2], json.loads(sys.argv[3])
    settings = spec.get('settings', {})
    if kind == 'pyspark':
        Pyspark.load_pyspark(name)(spec['source'], spec['destination'], settings, spec.get('properties', {}))
    elif kind == 'transformer':
        # transformers read only `work_path` from the config
        config = SimpleNamespace(work_path=Path(tempfile.mkdtemp(prefix='pts_')))
        Transform.load_transformer(name)(spec['source'], spec['destination'], settings, config)
    else:
        raise ValueError(f'unknown task kind {kind!r}, expected transformer or pyspark')


if __name__ == '__main__':
    main()
