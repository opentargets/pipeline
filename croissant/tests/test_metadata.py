"""Tests for the release-level Croissant metadata."""

import json
from datetime import datetime
from pathlib import Path

import mlcroissant as mlc
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from ot_croissant.crumbs.metadata import PlatformOutputMetadata

LOCATIONS = {
    'ftp_location': 'https://ftp.ebi.ac.uk/pub/databases/opentargets/platform/26.12/output/',
    'gcp_location': 'gs://open-targets-data-releases/26.12/output/',
    'aws_location': 's3://open-targets-public-data-releases/platform/26.12/output/',
}


@pytest.fixture
def dataset(tmp_path: Path) -> str:
    path = tmp_path / 'target'
    path.mkdir()
    pq.write_table(pa.table({'id': ['ENSG1']}), path / 'part-00000.parquet')
    return str(path)


def _metadata(dataset: str, **kwargs) -> PlatformOutputMetadata:
    return PlatformOutputMetadata(
        datasets=[dataset],
        version='26.12',
        date_published=datetime(2026, 12, 10),
        **LOCATIONS,
        **kwargs,
    )


def _locations(jsonld: dict) -> list[dict]:
    return [node for node in jsonld['distribution'] if node['@type'] == 'cr:FileObject']


def test_without_hash_is_live_and_carries_no_sha256(dataset: str) -> None:
    jsonld = _metadata(dataset).to_json()

    assert jsonld['isLiveDataset'] is True
    assert [node['@id'] for node in _locations(jsonld)] == ['ftp-location', 'gcp-location', 'aws-location']
    assert all('sha256' not in node for node in _locations(jsonld))


def test_without_hash_round_trips_through_mlcroissant(dataset: str, tmp_path: Path) -> None:
    path = tmp_path / 'croissant.json'
    path.write_text(json.dumps(_metadata(dataset).to_json(), default=str))

    reloaded = mlc.Dataset(str(path)).metadata

    assert not reloaded.ctx.issues.errors
    assert reloaded.ctx.is_live_dataset


def test_with_hash_is_not_live(dataset: str) -> None:
    jsonld = _metadata(dataset, data_integrity_hash='ab' * 32).to_json()

    assert 'isLiveDataset' not in jsonld
    assert {node['sha256'] for node in _locations(jsonld)} == {'ab' * 32}
