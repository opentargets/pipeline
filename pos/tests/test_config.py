"""The one config: its steps, no per-release values, and the paths its tasks share."""

import re
import sys
from pathlib import Path

import pytest
import yaml
from loguru import logger
from ot_datamodels.discovery import discover_models

from pos import core
from pos.clickhouse_ddl import input_dir, served_sql, served_views
from pos.run import PRODUCTS, parse_release_uri
from pos.tasks.clickhouse_load import ClickhouseLoadSpec
from pos.tasks.clickhouse_load_views import ClickhouseLoadViewsError, ClickhouseLoadViewsSpec, load_spec, view_loads

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
#: the served views
CLICKHOUSE = served_views()
SERVED_LOADS = 'clickhouse_load_views load served views'


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
        if task_type == 'clickhouse_load' and task['name'].split(' ')[0] == 'clickhouse_load_views':
            loaded |= set(served_views()) - set(task.get('exclude', []))
        for spec in [task, *task.get('do', [])]:
            if spec['name'].split(' ')[0] == task_type:
                loaded |= set(task['foreach']) if spec['dataset'] == '${each}' else {spec['dataset']}
    return loaded


def test_every_dataset_loads_in_its_servers_step():
    assert set(CLICKHOUSE) <= _loaded('backend_clickhouse', 'clickhouse_load')
    assert set(DATASETS['opensearch']) <= _loaded('backend_opensearch', 'open_search_load')
    assert not set(CLICKHOUSE) & _loaded('backend_opensearch', 'open_search_load')


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
    """Every `clickhouse_load` spec of the step as the registry builds it, served-view children as their task makes
    them."""
    built = []
    for spec in runner.specs:
        top = runner.task_registry.build(spec).spec
        if isinstance(top, ClickhouseLoadViewsSpec):
            built += [runner.task_registry.build(load_spec(top, v)).spec for v in view_loads(top.product, top.exclude)]
        else:
            built.append(top)
    return [s for s in built if isinstance(s, ClickhouseLoadSpec)]


def test_every_clickhouse_load_builds_with_the_product(load):
    _load, _ = load
    for run, product in (('platform-2612-3', 'platform'), ('ppp-2612-1', 'ppp')):
        runner, _ = _load('backend_clickhouse', run=run)
        loads = _built_loads(runner)
        # the served views, colocalisation included
        assert len(loads) == {'platform': 83, 'ppp': 91}[product]
        assert {s.product for s in loads} == {product}


@pytest.mark.parametrize('where', ['top', 'served'])
def test_a_clickhouse_load_without_the_product_fails(load, tmp_path, where):
    config = yaml.safe_load((ROOT / 'config' / 'config.yaml').read_text())
    tasks = config['steps']['backend_clickhouse']
    prefix = 'clickhouse_load ' if where == 'top' else 'clickhouse_load_views '
    first = next(t for t in tasks if t['name'].startswith(prefix))
    del first['product']
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    _load, lines = load
    with pytest.raises(SystemExit):
        _built_loads(_load('backend_clickhouse', config=path)[0])
    assert any('product: Field required' in line for line in lines)


def test_every_served_view_loads_from_its_view_folder():
    models = discover_models()
    for name, model in CLICKHOUSE.items():
        assert models[name] is model
        assert input_dir(model) == f'view/{name}'


def test_no_hand_written_clickhouse_sql():
    found = sorted(p.name for p in (ROOT / 'config' / 'clickhouse').iterdir() if not p.name.startswith('.'))
    assert found == ['config.d', 'users.d']


def test_every_load_waits_for_the_served_views_or_is_their_task():
    """Colocalisation, the one load outside the served views' task, requires it, so it loads alone after it."""
    tasks = {t['name']: t for t in CONFIG['steps']['backend_clickhouse']}
    assert not [name for name in tasks if name.startswith('explode ')]
    for name, task in tasks.items():
        if name.startswith('clickhouse_load '):
            assert task.get('requires') == [SERVED_LOADS], name


#: The served views each product loads in the served views' task: every view of the product but colocalisation.
VIEW_LOADS = [
    'association_otf_disease_view', 'association_otf_target_view', 'baseline_expression_view',
    'biosample_view', 'clinical_indication_by_disease_view', 'clinical_indication_by_drug_view',
    'clinical_report_view', 'clinical_target_by_target_view', 'coding_variant_by_target_view',
    'coding_variant_by_variant_view', 'credible_set_locus_view', 'credible_set_view', 'disease_hpo_view',
    'disease_phenotype_by_disease_view', 'disease_view', 'drug_mechanism_of_action_by_drug_view', 'drug_view',
    'drug_warning_by_drug_view', 'enhancer_to_gene_view',
    'evidence_cancer_biomarkers_by_disease_and_target_view', 'evidence_cancer_biomarkers_view',
    'evidence_cancer_gene_census_by_disease_and_target_view', 'evidence_cancer_gene_census_view',
    'evidence_clingen_by_disease_and_target_view', 'evidence_clingen_view',
    'evidence_clinical_precedence_by_disease_and_target_view', 'evidence_clinical_precedence_view',
    'evidence_crispr_by_disease_and_target_view', 'evidence_crispr_screen_by_disease_and_target_view',
    'evidence_crispr_screen_view', 'evidence_crispr_view',
    'evidence_europepmc_by_disease_and_target_view', 'evidence_europepmc_view',
    'evidence_eva_by_disease_and_target_view', 'evidence_eva_by_variant_view',
    'evidence_eva_somatic_by_disease_and_target_view', 'evidence_eva_somatic_by_variant_view',
    'evidence_eva_somatic_view', 'evidence_eva_view',
    'evidence_expression_atlas_by_disease_and_target_view', 'evidence_expression_atlas_view',
    'evidence_gene2phenotype_by_disease_and_target_view', 'evidence_gene2phenotype_view',
    'evidence_gene_burden_by_disease_and_target_view', 'evidence_gene_burden_view',
    'evidence_genomics_england_by_disease_and_target_view', 'evidence_genomics_england_view',
    'evidence_gwas_credible_sets_by_disease_and_target_view', 'evidence_gwas_credible_sets_view',
    'evidence_impc_by_disease_and_target_view', 'evidence_impc_view',
    'evidence_intogen_by_disease_and_target_view', 'evidence_intogen_view',
    'evidence_orphanet_by_disease_and_target_view', 'evidence_orphanet_view',
    'evidence_reactome_by_disease_and_target_view', 'evidence_reactome_view',
    'evidence_uniprot_literature_by_disease_and_target_view', 'evidence_uniprot_literature_view',
    'evidence_uniprot_variants_by_disease_and_target_view', 'evidence_uniprot_variants_by_variant_view',
    'evidence_uniprot_variants_view',
    'go_view', 'interaction_evidence_view', 'interaction_view',
    'l2g_prediction_by_credible_set_view', 'literature_entity_lut_view', 'literature_vector_view',
    'mouse_phenotype_by_target_view', 'pharmacogenomics_view', 'pharmacovigilance_by_drug_view', 'so_view',
    'study_credible_set_view', 'study_view', 'target_by_region_view', 'target_credible_set_view',
    'target_essentiality_view', 'target_prioritisation_view', 'target_view', 'transcript_view',
    'variant_credible_set_view', 'variant_view',
]  # fmt: skip
#: The PPP-only views.
PPP_ONLY = [
    'association_timeseries_view', 'otar_view',
    'evidence_encore_by_disease_and_target_view', 'evidence_encore_view',
    'evidence_ot_crispr_by_disease_and_target_view', 'evidence_ot_crispr_view',
    'evidence_ot_crispr_validation_by_disease_and_target_view', 'evidence_ot_crispr_validation_view',
]  # fmt: skip


def test_the_served_views_each_product_loads(load):
    _load, _ = load
    expected = {'platform': VIEW_LOADS, 'ppp': sorted([*VIEW_LOADS, *PPP_ONLY])}
    for run, product in (('platform-2612-3', 'platform'), ('ppp-2612-1', 'ppp')):
        runner, _ = _load('backend_clickhouse', run=run)
        (top,) = [s for s in (runner.task_registry.build(s).spec for s in runner.specs) if s.name == SERVED_LOADS]
        assert isinstance(top, ClickhouseLoadViewsSpec)
        assert view_loads(top.product, top.exclude) == expected[product]
        assert len(served_views(product)) == {'platform': 83, 'ppp': 91}[product]
        loaded = {s.dataset for s in _built_loads(runner)}
        assert loaded == set(served_views(product))


def test_served_loads_exclude_only_served_views():
    with pytest.raises(ClickhouseLoadViewsError, match=r"not served: \['nope'\]"):
        view_loads('platform', ['nope'])


def test_a_view_loads_only_for_its_products(load):
    _load, _ = load
    for run, product in (('platform-2612-3', 'platform'), ('ppp-2612-1', 'ppp')):
        loaded = {s.dataset for s in _built_loads(_load('backend_clickhouse', run=run)[0])}
        for name in loaded:
            assert product in CLICKHOUSE[name].products, f'{name} loads for {product}'
        assert all((name in loaded) == (product == 'ppp') for name in PPP_ONLY)


def test_served_table_and_dictionary_names_are_unique():
    """A table or dictionary name served twice would drop the first one's table at load time."""
    names = []
    for name, model in CLICKHOUSE.items():
        assert model.clickhouse is not None
        names += [name] + [d.name for d in model.clickhouse.dictionaries]
        assert served_sql(model)
    assert len(names) == len(set(names)), sorted(n for n in set(names) if names.count(n) > 1)
