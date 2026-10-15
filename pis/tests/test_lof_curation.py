"""Download configuration and partition-preserving copy integration for lof_curation."""

from inspect import unwrap
from pathlib import Path

import yaml
from otter.config.model import Config
from otter.scratchpad.model import Scratchpad
from otter.task.model import TaskContext
from otter.tasks.copy import Copy, CopySpec
from otter.tasks.explode_glob import ExplodeGlob, ExplodeGlobSpec

STEP = 'lof_curation'
SOURCE = 'gs://otar013-ppp/OTAR2075_lof_curation/lof_curation_variant_annotations/**/*.parquet'
DESTINATION = 'input/variant/lof_curation'
ROOT = Path(__file__).parents[2]


def configured_task():
    config = yaml.safe_load((ROOT / 'pis/config.yaml').read_text())
    tasks = config['steps'][STEP]
    assert len(tasks) == 1
    return tasks[0]


def test_download_configuration_and_pipeline_registration():
    task = configured_task()
    spec = ExplodeGlobSpec.model_validate(task)
    assert spec.glob == SOURCE
    copy = CopySpec.model_validate(spec.do[0].model_dump())
    assert copy.source == '${uri}'
    assert copy.destination == DESTINATION + '/${match_path}/${match_stem}.${match_ext}'
    pipeline = yaml.safe_load((ROOT / 'orchestration/src/orchestration/dags/config/unified_pipeline.yaml').read_text())
    assert 'pis_' + STEP in pipeline['steps']
    assert bool((pipeline['steps']['pis_' + STEP] or {}).get('ppp_only')) is True


def test_download_preserves_flat_and_nested_parquet_paths(tmp_path):
    """Same-named parts in different partitions survive; non-Parquet objects are ignored."""
    source = tmp_path / 'source'
    expected = {
        'part.parquet': b'flat partition',
        'chromosome=1/part.parquet': b'chromosome one',
        'chromosome=2/nested/part.parquet': b'chromosome two',
    }
    for name, content in expected.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    (source / '_SUCCESS').write_text('')
    (source / 'not-a-part.csv').write_text('ignore this')
    task = configured_task()
    task['glob'] = str(source / '**/*.parquet')
    config = Config.model_construct(step=STEP, work_path=tmp_path / 'work', release_uri=None)
    context = TaskContext(config, Scratchpad())
    explode = ExplodeGlob(ExplodeGlobSpec.model_validate(task), context)
    unwrap(ExplodeGlob.run)(explode)
    assert len(context.specs) == len(expected)
    for nested in context.specs:
        copy = Copy(CopySpec.model_validate(nested.model_dump()), context)
        unwrap(Copy.run)(copy)
    output = tmp_path / 'work' / DESTINATION
    actual = {str(path.relative_to(output)): path.read_bytes() for path in output.rglob('*') if path.is_file()}
    assert actual == expected
