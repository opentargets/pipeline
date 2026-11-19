"""BigQuery staging and swap, against a fake client that records calls."""

from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml
from otter.task.model import Spec

from pos.gcp.bigquery import BigQueryError, BigQueryRelease, published_tables
from pos.run import for_product

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
    _release(client).publish(published_tables(uri), uri)
    (load,) = client.made('load')
    assert load[1:3] == (f'{uri}/output/mouse_phenotype/*.parquet', f'{PROJECT}.{STAGING}.mouse_phenotype')
    assert load[3:] == ('WRITE_TRUNCATE', 'PARQUET')
    assert client.prod() == {'mouse_phenotype'}


def test_failed_load_never_touches_prod():
    client = FakeClient(prod=['disease', 'known_drug'], fail_load=['go'])
    with pytest.raises(BigQueryError, match=r'\bgo\b') as e:
        _release(client).publish(['disease', 'go', 'so'], 'gs://b/r')
    assert 'disease' not in str(e.value)
    assert len(client.made('load')) == 3  # every job awaited before raising
    assert client.made('copy') == client.made('delete') == []
    assert client.prod() == {'disease', 'known_drug'}


def test_empty_table_blocks_swap():
    client = FakeClient(prod=['disease'], rows={'so': 0})
    with pytest.raises(BigQueryError, match=r'\bso\b'):
        _release(client).publish(['disease', 'so'], 'gs://b/r')
    assert client.made('copy') == client.made('delete') == []


def test_swap_copies_and_prunes():
    client = FakeClient(prod=['disease', 'known_drug'])
    _release(client).publish(['disease', 'go'], 'gs://b/r')
    copies = [c[1:] for c in client.made('copy')]
    assert copies == [
        (f'{PROJECT}.{STAGING}.{t}', f'{PROJECT}.{PROD}.{t}', 'WRITE_TRUNCATE') for t in ['disease', 'go']
    ]
    assert client.made('delete') == [('delete', f'{PROJECT}.{PROD}.known_drug')]
    # prune follows every copy
    kinds = [c[0] for c in client.calls]
    assert kinds.index('delete') > max(i for i, k in enumerate(kinds) if k == 'copy')
    assert client.prod() == {'disease', 'go'}


def test_swap_keeps_views():
    client = FakeClient(prod=['disease'])
    client.tables[f'{PROJECT}.{PROD}.a_view'] = _Table('a_view', 0, 'VIEW')
    _release(client).publish(['disease'], 'gs://b/r')
    assert client.made('delete') == []


def test_swap_rerun_idempotent():
    tables = ['disease', 'go', 'so']
    client = FakeClient(prod=['disease', 'known_drug'], fail_copy=['go'])
    with pytest.raises(BigQueryError, match=r'\bgo\b'):
        _release(client).publish(tables, 'gs://b/r')
    first = [c for c in client.calls if c[0] in ('load', 'copy')]
    assert len(client.made('copy')) == 3  # the other copies still ran
    assert client.made('delete') == []
    assert 'known_drug' in client.prod()

    client.fail_copy.clear()
    client.calls.clear()
    _release(client).publish(tables, 'gs://b/r')
    assert [c for c in client.calls if c[0] in ('load', 'copy')] == first
    assert client.made('delete') == [('delete', f'{PROJECT}.{PROD}.known_drug')]
    assert client.prod() == set(tables)


def test_no_tables_refused():
    client = FakeClient(prod=['disease'])
    with pytest.raises(BigQueryError, match='no tables'):
        _release(client).publish([], 'gs://b/r')
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
        'project_id': '${bigquery_project}',
        'dataset_id': '${bigquery_dataset}',
        'staging_dataset_id': '${bigquery_staging_dataset}',
        'location': '${bigquery_location}',
    }
