"""BigQuery staging and swap, against a fake client that records calls."""

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from otter.task.model import Spec
from otter.util.errors import OtterError

from pos.gcp.bigquery import BigQueryError, BigQueryRelease, published_tables
from pos.run import for_product
from pos.tasks.bigquery_drop_datasets import BigqueryDropDatasets, BigqueryDropDatasetsSpec

CONFIG = yaml.safe_load((Path(__file__).parents[1] / 'config' / 'config.yaml').read_text())
PROJECT, PROD, STAGING = 'p', 'platform', 'platform_staging'


@dataclass
class _Table:
    table_id: str
    num_rows: int
    table_type: str = 'TABLE'


class _Job:
    def __init__(self, error: str | None, done):
        self._error, self._done = error, done

    def result(self):
        if self._error:
            raise RuntimeError(self._error)
        self._done()
        return self


class FakeClient:
    """Tables by full id; jobs apply on `result()`, failing for the scripted table names."""

    def __init__(self, prod=(), rows=None, fail_load=(), fail_copy=()):
        self.tables = {f'{PROJECT}.{PROD}.{t}': _Table(t, 1) for t in prod}
        self.rows = rows or {}
        self.fail_load, self.fail_copy = set(fail_load), set(fail_copy)
        self.calls: list[tuple] = []

    def create_dataset(self, dataset, exists_ok=False):
        self.calls.append(('create_dataset', dataset.dataset_id))

    def load_table_from_uri(self, uri, destination, job_config, location=None):
        self.calls.append(('load', uri, destination, job_config.write_disposition, job_config.source_format))
        name = destination.rsplit('.', 1)[-1]
        rows = self.rows.get(name, 1)
        return self._job(name in self.fail_load, destination, lambda: self._put(destination, rows))

    def load_table_from_json(self, rows, destination, job_config, location=None):
        rows = list(rows)
        schema = [(f.name, f.field_type) for f in job_config.schema]
        self.calls.append(('load_json', rows, destination, job_config.write_disposition, schema))
        name = destination.rsplit('.', 1)[-1]
        return self._job(name in self.fail_load, destination, lambda: self._put(destination, len(rows)))

    def copy_table(self, source, destination, job_config, location=None):
        self.calls.append(('copy', source, destination, job_config.write_disposition))
        name = destination.rsplit('.', 1)[-1]
        return self._job(name in self.fail_copy, destination, lambda: self._copy(source, destination))

    def get_table(self, table):
        return self.tables[table]

    def list_tables(self, dataset):
        return [t for k, t in sorted(self.tables.items()) if k.startswith(f'{dataset}.')]

    def delete_table(self, table, not_found_ok=False):
        self.calls.append(('delete', table))
        self.tables.pop(table, None)

    def _copy(self, source, destination):
        self._put(destination, self.tables[source].num_rows)

    def _put(self, table, rows):
        self.tables[table] = _Table(table.rsplit('.', 1)[-1], rows)

    def _job(self, fails, destination, done):
        return _Job(f'job into {destination} failed' if fails else None, done)

    def prod(self) -> set[str]:
        return {k.rsplit('.', 1)[-1] for k in self.tables if k.startswith(f'{PROJECT}.{PROD}.')}

    def made(self, kind) -> list[tuple]:
        return [c for c in self.calls if c[0] == kind]


def _release(client) -> BigQueryRelease:
    return BigQueryRelease(client, project=PROJECT, dataset=PROD, staging=STAGING, location='eu')


def _output(root: Path, *names: str) -> str:
    for name in names:
        (root / 'output' / name).mkdir(parents=True)
        (root / 'output' / name / 'part-0.parquet').write_bytes(b'')
    return str(root)


def test_published_tables(tmp_path):
    uri = _output(tmp_path, 'disease', 'mouse_phenotype', 'not_a_model')
    (tmp_path / 'output' / 'target').mkdir()  # no parquet
    assert published_tables(uri) == ['disease', 'mouse_phenotype']


def test_mouse_phenotype_name(tmp_path):
    uri = _output(tmp_path, 'mouse_phenotype')
    client = FakeClient()
    _release(client).publish(published_tables(uri), uri, '26.12')
    (load,) = client.made('load')
    assert load[1:3] == (f'{uri}/output/mouse_phenotype/*.parquet', f'{PROJECT}.{STAGING}.mouse_phenotype')
    assert load[3:] == ('WRITE_TRUNCATE', 'PARQUET')
    assert client.prod() == {'mouse_phenotype', 'ot_release'}


def test_failed_load_never_touches_prod():
    client = FakeClient(prod=['disease', 'known_drug'], fail_load=['go'])
    with pytest.raises(BigQueryError, match=r'\bgo\b') as e:
        _release(client).publish(['disease', 'go', 'so'], 'gs://b/r', '26.12')
    assert 'disease' not in str(e.value)
    assert len(client.made('load')) == 3  # every job awaited before raising
    assert client.made('copy') == client.made('delete') == []
    assert client.prod() == {'disease', 'known_drug'}


def test_empty_table_blocks_swap():
    client = FakeClient(prod=['disease'], rows={'so': 0})
    with pytest.raises(BigQueryError, match=r'\bso\b'):
        _release(client).publish(['disease', 'so'], 'gs://b/r', '26.12')
    assert client.made('copy') == client.made('delete') == []


def test_swap_copies_and_prunes():
    client = FakeClient(prod=['disease', 'known_drug', 'ot_release'])
    _release(client).publish(['disease', 'go'], 'gs://b/r', '26.12')
    copies = [c[1:] for c in client.made('copy')]
    assert copies == [
        (f'{PROJECT}.{STAGING}.{t}', f'{PROJECT}.{PROD}.{t}', 'WRITE_TRUNCATE') for t in ['disease', 'go', 'ot_release']
    ]
    assert client.made('delete') == [('delete', f'{PROJECT}.{PROD}.known_drug')]
    # prune follows every copy
    kinds = [c[0] for c in client.calls]
    assert kinds.index('delete') > max(i for i, k in enumerate(kinds) if k == 'copy')
    assert client.prod() == {'disease', 'go', 'ot_release'}


def test_ot_release_written_and_swapped():
    client = FakeClient(prod=['ot_release'])
    _release(client).publish(['disease'], 'gs://b/r', '26.12')
    (load,) = client.made('load_json')
    assert load[1:] == (
        [{'release': '26.12'}],
        f'{PROJECT}.{STAGING}.ot_release',
        'WRITE_TRUNCATE',
        [('release', 'STRING')],
    )
    assert (f'{PROJECT}.{STAGING}.ot_release', f'{PROJECT}.{PROD}.ot_release') in [c[1:3] for c in client.made('copy')]
    assert client.made('delete') == []


def test_swap_keeps_views():
    client = FakeClient(prod=['disease'])
    client.tables[f'{PROJECT}.{PROD}.a_view'] = _Table('a_view', 0, 'VIEW')
    _release(client).publish(['disease'], 'gs://b/r', '26.12')
    assert client.made('delete') == []


def test_swap_rerun_idempotent():
    tables = ['disease', 'go', 'so']
    client = FakeClient(prod=['disease', 'known_drug'], fail_copy=['go'])
    with pytest.raises(BigQueryError, match=r'\bgo\b'):
        _release(client).publish(tables, 'gs://b/r', '26.12')
    first = [c for c in client.calls if c[0] in ('load', 'copy')]
    assert len(client.made('copy')) == 4  # the other copies still ran
    assert client.made('delete') == []
    assert 'known_drug' in client.prod()

    client.fail_copy.clear()
    client.calls.clear()
    _release(client).publish(tables, 'gs://b/r', '26.12')
    assert [c for c in client.calls if c[0] in ('load', 'copy')] == first
    assert client.made('delete') == [('delete', f'{PROJECT}.{PROD}.known_drug')]
    assert client.prod() == {*tables, 'ot_release'}


def test_no_tables_refused():
    client = FakeClient(prod=['disease'])
    with pytest.raises(BigQueryError, match='no tables'):
        _release(client).publish([], 'gs://b/r', '26.12')
    assert client.calls == []


def test_bigquery_refuses_ppp():
    specs = [Spec(**s) for s in CONFIG['steps']['bigquery']]
    with pytest.raises(ValueError, match='step bigquery does not run for ppp'):
        for_product(specs, 'ppp', 'bigquery')
    assert [s.name for s in for_product(specs, 'platform', 'bigquery')] == ['bigquery_load release tables']


def test_bigquery_step_fields():
    (spec,) = CONFIG['steps']['bigquery']
    assert spec == {
        'name': 'bigquery_load release tables',
        'products': ['platform'],
        'release_uri': '${data_source}',
        'release': '${release}',
        'project_id': '${bigquery_project}',
        'dataset_id': '${bigquery_dataset}',
        'staging_dataset_id': '${bigquery_staging_dataset}',
        'location': '${bigquery_location}',
    }


def _drop(mocker, project_id, dataset_ids):
    client = mocker.patch('pos.tasks.bigquery_drop_datasets.bigquery.Client')
    spec = BigqueryDropDatasetsSpec(name='bigquery_drop_datasets x', project_id=project_id, dataset_ids=dataset_ids)
    task = BigqueryDropDatasets(spec, SimpleNamespace())  # ty: ignore[invalid-argument-type]
    task.run.__wrapped__(task)  # ty: ignore[unresolved-attribute]
    return client


def test_drop_refuses_prod(mocker):
    client = mocker.patch('pos.tasks.bigquery_drop_datasets.bigquery.Client')
    with pytest.raises(OtterError, match='open-targets-prod'):
        _drop(mocker, 'open-targets-prod', ['platform_scratch'])
    client.assert_not_called()


def test_drop_refuses_non_scratch_names(mocker):
    client = mocker.patch('pos.tasks.bigquery_drop_datasets.bigquery.Client')
    with pytest.raises(OtterError, match='platform'):
        _drop(mocker, 'open-targets-eu-dev', ['platform_scratch', 'platform'])
    client.assert_not_called()


def test_drop_deletes_both(mocker):
    client = _drop(mocker, 'open-targets-eu-dev', ['platform_scratch', 'platform_scratch_staging'])
    client.assert_called_once_with(project='open-targets-eu-dev')
    delete = client.return_value.delete_dataset
    assert delete.call_args_list == [
        mocker.call('open-targets-eu-dev.platform_scratch', delete_contents=True, not_found_ok=True),
        mocker.call('open-targets-eu-dev.platform_scratch_staging', delete_contents=True, not_found_ok=True),
    ]
