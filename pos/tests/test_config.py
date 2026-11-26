"""The one config: its steps, no per-release values, and the paths its tasks share."""

import re
import sys
from fnmatch import fnmatch
from pathlib import Path

import pytest
import yaml
from loguru import logger
from ot_datamodels.discovery import discover_models
from otter.scratchpad.model import Scratchpad
from otter.tasks.explode import ExplodeSpec

from pos import core
from pos.clickhouse_ddl import declarations, entry_model, is_declared
from pos.run import PRODUCTS, parse_release_uri
from pos.tasks.clickhouse_load import ClickhouseLoadSpec

ROOT = Path(__file__).parents[1]
CONFIG = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())
RELEASE_NUMBER = re.compile(r'(?<![\d.])\d{2}\.?\d{2}(?![\d.])')
# derived from release_uri by pos.run
DERIVED = set(parse_release_uri('gs://b/platform-2612-1').scratchpad()).union(*PRODUCTS.values())
BACKEND_GUARDS = {
    'backend_clickhouse': 'clickhouse_create_database release database',
    'backend_opensearch': 'open_search_prepare cluster settings',
}
DATASETS = yaml.safe_load((ROOT / 'config' / 'datasets.yaml').read_text())


def _specs(tasks: list[dict]) -> list[dict]:
    specs = []
    for task in tasks:
        specs.append(task)
        specs.extend(_specs(task.get('do', [])))
    return specs


def _strings(value) -> list[str]:
    if isinstance(value, str):
        return [value]
    if isinstance(value, dict):
        return [s for v in value.values() for s in _strings(v)]
    if isinstance(value, list):
        return [s for v in value for s in _strings(v)]
    return []


def _task(step: str, task_type: str) -> dict:
    (spec,) = [s for s in _specs(CONFIG['steps'][step]) if s['name'].split(' ')[0] == task_type]
    return spec


def test_config_steps():
    assert set(CONFIG['steps']) == {
        'backend_clickhouse', 'backend_opensearch', 'tarballs', 'bigquery', 'gcs_sync', 'aws_sync', 'ftp_sync',
        'publish_manifest', 'scratch_cleanup',
    }  # fmt: skip


def test_ppp_config_is_gone():
    assert not (ROOT / 'config' / 'config_ppp.yaml').exists()


def test_no_hand_edited_fields():
    assert 'release_uri' not in CONFIG
    assert not DERIVED & set(CONFIG['scratchpad'])
    for value in _strings(CONFIG['scratchpad']):
        assert not RELEASE_NUMBER.search(value), value


def test_no_disk_or_snapshot_fields():
    text = (ROOT / 'config' / 'config.yaml').read_text()
    assert 'snapshot' not in text
    assert 'disk_name' not in text


def test_platform_only_steps():
    for step in ('bigquery', 'aws_sync', 'ftp_sync'):
        assert all(s.get('products') == ['platform'] for s in CONFIG['steps'][step]), step
    for step in ('backend_clickhouse', 'backend_opensearch', 'tarballs', 'gcs_sync'):
        assert not any('products' in s and 'ppp' not in s['products'] for s in CONFIG['steps'][step]), step


def test_backend_steps():
    assert 'backend' not in CONFIG['steps']
    assert {'backend_clickhouse', 'backend_opensearch'} <= set(CONFIG['steps'])


def test_backend_steps_use_one_server_each():
    for s in _specs(CONFIG['steps']['backend_clickhouse']):
        assert not s['name'].startswith(('open_search_', 'ot_croissant')), s['name']
    for s in _specs(CONFIG['steps']['backend_opensearch']):
        assert not s['name'].startswith('clickhouse_'), s['name']


def _sync_sources(step: str) -> set[str]:
    return {s['source'] for s in _specs(CONFIG['steps'][step]) if s['name'].startswith('sync_bucket ')}


def test_backend_step_syncs():
    assert _sync_sources('backend_clickhouse') == {'${data_source}/output', '${data_source}/view'}
    assert _sync_sources('backend_opensearch') == {'${data_source}/output', '${data_source}/export'}


def test_croissant_only_in_opensearch_step():
    name = 'ot_croissant generate release metadata'
    assert name in {s['name'] for s in CONFIG['steps']['backend_opensearch']}
    assert name not in {s['name'] for s in _specs(CONFIG['steps']['backend_clickhouse'])}


def _loaded(step: str, task_type: str) -> set[str]:
    loaded = set()
    for task in CONFIG['steps'][step]:
        for spec in [task, *task.get('do', [])]:
            if spec['name'].split(' ')[0] == task_type:
                loaded |= set(task['foreach']) if spec['dataset'] == '${each}' else {spec['dataset']}
    return loaded


def test_every_dataset_loads_in_its_servers_step():
    assert set(DATASETS['clickhouse']) <= _loaded('backend_clickhouse', 'clickhouse_load')
    assert set(DATASETS['opensearch']) <= _loaded('backend_opensearch', 'open_search_load')
    assert not set(DATASETS['clickhouse']) & _loaded('backend_opensearch', 'open_search_load')


def test_croissant_lands_where_opensearch_loads_it():
    croissant = _task('backend_opensearch', 'ot_croissant')
    load = _task('backend_opensearch', 'open_search_load')
    assert croissant['prepared_data_parent'] == load['json_parent']


def test_derived_fields_cover_the_run():
    assert {'release', 'data_source', 'bigquery_project', 'ftp_address', 'croissant_ftp_address'} <= DERIVED


def test_backend_syncs_wait_for_the_rerun_guards():
    for step, guard in BACKEND_GUARDS.items():
        top = {s['name'] for s in CONFIG['steps'][step]}
        assert guard in top
        syncs = [s for s in CONFIG['steps'][step] if s['name'].startswith('sync_bucket ')]
        assert len(syncs) == 2
        assert all(s['requires'] == [guard] for s in syncs), step


@pytest.fixture
def load(tmp_path, monkeypatch):
    """Set up the runner for a step of the real config: its specs as the registry builds them, and the log."""
    monkeypatch.setattr('otter.core.init_logger_early', lambda *_: None)
    monkeypatch.setattr('otter.core.init_logger', lambda *_: None)
    lines: list[str] = []
    sink = logger.add(lambda m: lines.append(str(m)), level='INFO')

    def _load(step: str, run: str = 'platform-2612-3', scratch: bool = False, config: Path | None = None):
        if scratch:
            monkeypatch.setenv('POS_SCRATCH', '1')
        else:
            monkeypatch.delenv('POS_SCRATCH', raising=False)
        uri = f'gs://open-targets-pipeline-runs/do/{run}'
        config = config or ROOT / 'config' / 'config.yaml'
        argv = ['pos', '-s', step, '-c', str(config), '-w', str(tmp_path / 'work'), '-r', uri]
        monkeypatch.setattr(sys, 'argv', argv)
        runner = core.setup()
        return runner, [runner.task_registry.build(s).spec for s in runner.specs]

    yield _load, lines
    logger.remove(sink)


def test_publish_manifest_and_scratch_cleanup_resolve(load, mocker):
    _load, lines = load
    _, specs = _load('publish_manifest')
    assert [(s.source, s.destination) for s in specs] == [
        ('gs://open-targets-pipeline-runs/do/platform-2612-3/manifest.json',
         'gs://open-targets-data-releases/26.12/manifest.json'),
        ('gs://open-targets-pipeline-runs/do/platform-2612-3/manifest.json',
         's3://open-targets-public-data-releases/platform/26.12/manifest.json'),
    ]  # fmt: skip

    _, specs = _load('publish_manifest', scratch=True)
    assert [(s.source, s.destination) for s in specs] == [
        ('gs://open-targets-pos-scratch/runs/do-platform-2612-3/manifest.json',
         'gs://open-targets-pos-scratch/platform/26.12/manifest.json'),
    ]  # fmt: skip

    with pytest.raises(SystemExit):
        _load('aws_sync', scratch=True)
    assert any('step aws_sync does not run in scratch mode' in line for line in lines)

    with pytest.raises(SystemExit):
        _load('scratch_cleanup')
    assert any('step scratch_cleanup does not run in normal mode' in line for line in lines)

    _, specs = _load('scratch_cleanup', scratch=True)
    assert [(s.project_id, s.dataset_ids) for s in specs] == [
        ('open-targets-eu-dev', ['platform_scratch', 'platform_scratch_staging']),
    ]

    runner, specs = _load('scratch_cleanup', run='ppp-2612-1', scratch=True)
    assert specs == []
    run = mocker.patch.object(core.Runner, 'run')
    mocker.patch.object(core, 'setup', return_value=runner)
    core.main()
    run.assert_not_called()
    assert any('step scratch_cleanup does not apply to ppp' in line for line in lines)


def test_ftp_and_aws_skip_scratch():
    for step in ('aws_sync', 'ftp_sync'):
        assert all(s.get('scratch') == 'skip' for s in CONFIG['steps'][step]), step


def test_every_clickhouse_load_passes_the_product():
    loads = [s for s in _specs(CONFIG['steps']['backend_clickhouse']) if s['name'].startswith('clickhouse_load ')]
    assert loads
    assert all(s.get('product') == '${product}' for s in loads)


def _built_loads(runner) -> list[ClickhouseLoadSpec]:
    """Every `clickhouse_load` spec of the step as the registry builds it, explode children as `explode` makes them."""
    built = []
    for spec in runner.specs:
        top = runner.task_registry.build(spec).spec
        if not isinstance(top, ExplodeSpec):
            built.append(top)
            continue
        for each in top.foreach:
            for child in top.do:
                pad = Scratchpad({})
                pad.store(top.each_placeholder, each)
                built.append(runner.task_registry.build(child.model_validate(pad.replace_dict(child.model_dump()))).spec)
    return [s for s in built if isinstance(s, ClickhouseLoadSpec)]


def test_every_clickhouse_load_builds_with_the_product(load):
    _load, _ = load
    for run, product in (('platform-2612-3', 'platform'), ('ppp-2612-1', 'ppp')):
        runner, _ = _load('backend_clickhouse', run=run)
        loads = _built_loads(runner)
        assert len(loads) > 30
        assert {s.product for s in loads} == {product}


@pytest.mark.parametrize('where', ['top', 'explode'])
def test_a_clickhouse_load_without_the_product_fails(load, tmp_path, where):
    config = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())
    tasks = config['steps']['backend_clickhouse']
    if where == 'explode':
        explode = next(t for t in tasks if t['name'].startswith('explode '))
        tasks = explode['do']
    first = next(t for t in tasks if t['name'].startswith('clickhouse_load '))
    del first['product']
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    _load, lines = load
    with pytest.raises(SystemExit):
        _built_loads(_load('backend_clickhouse', config=path)[0])
    assert any('product: Field required' in line for line in lines)


def test_clickhouse_entries_have_one_ddl_source():
    for name, entry in DATASETS['clickhouse'].items():
        sources = [k for k in ('model', 'models', 'preload_script') if k in entry]
        assert len(sources) == 1, f'{name}: {sources}'
    hand_written = [n for n, e in DATASETS['clickhouse'].items() if 'preload_script' in e and not e.get('derived')]
    assert not hand_written, f'landing tables with no model: {hand_written}'


def test_clickhouse_models_are_datasets():
    models = discover_models()
    for name, entry in DATASETS['clickhouse'].items():
        input_dir = entry.get('input_dir', '')
        for prefix in ('output/', 'view/'):
            if input_dir.startswith(prefix) and input_dir.removeprefix(prefix) in models:
                assert entry.get('model') == input_dir.removeprefix(prefix), f'{name}: {input_dir} has a model'
        if 'model' in entry:
            assert entry['model'] in models, name
        if 'models' in entry:
            assert any(fnmatch(m, entry['models']) for m in models), name
        for optional in entry.get('optional_models', []):
            assert optional in models and fnmatch(optional, entry.get('models', '')), f'{name}: {optional}'


def test_no_hand_written_landing_ddl():
    assert not list((ROOT / 'config' / 'clickhouse' / 'schema').glob('*'))
    for name, entry in DATASETS['clickhouse'].items():
        if 'preload_script' in entry:
            assert entry['preload_script'].startswith('config/clickhouse/scripts/'), name


DECLARED_LOADS = 'explode load declared datasets'


def test_undeclared_loads_wait_for_the_declared_ones():
    """Every load of an undeclared dataset requires the declared explode, directly or through another task; a
    declared dataset outside that explode is product-gated or waits for it too."""
    tasks = {t['name']: t for t in CONFIG['steps']['backend_clickhouse']}
    assert all(is_declared(DATASETS['clickhouse'][d]) for d in tasks[DECLARED_LOADS]['foreach'])

    def waits(name: str) -> bool:
        requires = tasks[name].get('requires', [])
        return DECLARED_LOADS in requires or any(waits(r) for r in requires if r in tasks)

    for name, task in tasks.items():
        if name == DECLARED_LOADS or not name.startswith(('clickhouse_load', 'explode ')):
            continue
        datasets = task.get('foreach', [task.get('dataset')])
        if all(is_declared(DATASETS['clickhouse'][d]) for d in datasets):
            assert task.get('products') or waits(name), f'{name}: a declared dataset outside {DECLARED_LOADS}'
            continue
        assert waits(name), f'{name} does not wait for {DECLARED_LOADS}'


def test_a_declared_model_loads_only_for_its_products(load):
    """A declared model builds no table for a product it lacks, so its load runs only for its products."""
    _load, _ = load
    for run, product in (('platform-2612-3', 'platform'), ('ppp-2612-1', 'ppp')):
        loaded = {s.dataset for s in _built_loads(_load('backend_clickhouse', run=run)[0])}
        for name in loaded:
            entry = DATASETS['clickhouse'][name]
            model = entry_model(entry)
            if model is not None and is_declared(entry):
                assert product in model.products, f'{name} loads for {product}, its model is {model.products}'
        assert ('otar_projects' in loaded) == (product == 'ppp')


@pytest.mark.parametrize('product', sorted(PRODUCTS))
def test_declared_names_are_unique(product):
    """A table or dictionary name declared twice would drop the first one's table at load time."""
    names = []
    for entry in DATASETS['clickhouse'].values():
        tables, dictionaries = declarations(entry, product)
        names += [t.name for t in tables] + [d.name for d in dictionaries]
    assert len(names) == len(set(names)), sorted(n for n in set(names) if names.count(n) > 1)
