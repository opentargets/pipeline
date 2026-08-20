"""Tests for ``UploadRemoteFileOperator``.

The operator stages a large remote artifact (the ~629 MB Spark-NLP fat jar) into
GCS once per cluster. Its two risky behaviours are ``skip_if_exists`` — which
must not trust an object just because the name matches — and cleanup of the
temporary download, which is big enough to fill a worker's disk across retries.
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from orchestration.operators.gcs import UploadRemoteFileOperator

DST = 'gs://opentargets-pipelines/up/pts/jars/spark-nlp-assembly-6.1.5.jar'
SRC = 'https://s3.amazonaws.com/auxdata.johnsnowlabs.com/public/jars/spark-nlp-assembly-6.1.5.jar'
PAYLOAD = b'jar-bytes'


@pytest.fixture
def gcs():
    """Mock the GCS client and bucket the operator talks to."""
    with (
        patch('orchestration.operators.gcs.Client', MagicMock()),
        patch('orchestration.operators.gcs.Bucket') as bucket_cls,
    ):
        bucket = MagicMock()
        bucket.exists.return_value = True
        bucket.get_blob.return_value = None
        bucket_cls.return_value = bucket
        yield bucket


@pytest.fixture
def http():
    """Mock ``requests`` while keeping its real exception types.

    The operator catches ``requests.RequestException``; a MagicMock in an except
    clause raises ``TypeError`` instead of being caught.
    """
    with patch('orchestration.operators.gcs.requests') as mock:
        mock.RequestException = requests.RequestException
        yield mock


def _operator(**kwargs) -> UploadRemoteFileOperator:
    return UploadRemoteFileOperator(task_id='stage_jar', src_url=SRC, dst_uri=DST, **kwargs)


def _staged(bucket: MagicMock, size: int) -> None:
    """Put an object of ``size`` bytes at the destination."""
    existing = MagicMock()
    existing.size = size
    bucket.get_blob.return_value = existing


def _download(http: MagicMock, body: bytes, content_length: int | None) -> None:
    r = MagicMock()
    r.__enter__.return_value = r
    r.headers = {} if content_length is None else {'Content-Length': str(content_length)}
    r.iter_content.return_value = [body]
    http.get.return_value = r


def _head(http: MagicMock, content_length: int) -> None:
    r = MagicMock()
    r.headers = {'Content-Length': str(content_length)}
    http.head.return_value = r


def test_uploads_when_destination_is_absent(gcs, http):
    """The ordinary cold-bucket path: download, verify size, upload."""
    _download(http, PAYLOAD, len(PAYLOAD))

    _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_skips_when_existing_object_matches_source_size(gcs, http):
    """A previously staged artifact of the right size is reused, not re-fetched."""
    _staged(gcs, len(PAYLOAD))
    _head(http, len(PAYLOAD))

    _operator(skip_if_exists=True).execute({})

    http.get.assert_not_called()
    gcs.blob.return_value.upload_from_filename.assert_not_called()


def test_restages_when_existing_object_is_the_wrong_size(gcs, http):
    """A stale or truncated object under the right name must not be trusted.

    The realistic trigger is a hand-staged jar of a different version sitting in
    the bucket under the name this run expects.
    """
    _staged(gcs, 999999)
    _head(http, len(PAYLOAD))
    _download(http, PAYLOAD, len(PAYLOAD))

    _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_keeps_existing_object_when_source_size_is_unknown(gcs, http):
    """An unreachable source is no reason to discard a good staged copy."""
    _staged(gcs, len(PAYLOAD))
    http.head.side_effect = requests.RequestException('network down')

    _operator(skip_if_exists=True).execute({})

    http.get.assert_not_called()
    gcs.blob.return_value.upload_from_filename.assert_not_called()


def test_skip_if_exists_disabled_always_downloads(gcs, http):
    """Without the flag the destination is overwritten unconditionally."""
    _staged(gcs, len(PAYLOAD))
    _download(http, PAYLOAD, len(PAYLOAD))

    _operator(skip_if_exists=False).execute({})

    http.head.assert_not_called()
    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_truncated_download_is_not_uploaded(gcs, http):
    """A short stream must fail loudly instead of publishing a corrupt artifact."""
    _download(http, PAYLOAD, len(PAYLOAD) + 500)

    with pytest.raises(ValueError, match='truncated'):
        _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_not_called()


def test_download_without_content_length_is_uploaded(gcs, http):
    """A source that advertises no size is still staged; there is nothing to check."""
    _download(http, PAYLOAD, None)

    _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_temp_file_is_removed_on_failure(gcs, http):
    """The ~629 MB download must not survive a failed attempt."""
    _download(http, PAYLOAD, len(PAYLOAD) + 500)

    created: list[Path] = []
    real_named_temp = tempfile.NamedTemporaryFile

    def _spy(*args, **kwargs):
        handle = real_named_temp(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    with patch('orchestration.operators.gcs.tempfile.NamedTemporaryFile', _spy):
        with pytest.raises(ValueError):
            _operator(skip_if_exists=True).execute({})

    assert created, 'expected the operator to create a temporary file'
    assert not any(p.exists() for p in created)


def test_temp_file_is_removed_on_success(gcs, http):
    """The same cleanup applies to the happy path."""
    _download(http, PAYLOAD, len(PAYLOAD))

    created: list[Path] = []
    real_named_temp = tempfile.NamedTemporaryFile

    def _spy(*args, **kwargs):
        handle = real_named_temp(*args, **kwargs)
        created.append(Path(handle.name))
        return handle

    with patch('orchestration.operators.gcs.tempfile.NamedTemporaryFile', _spy):
        _operator(skip_if_exists=True).execute({})

    assert created
    assert not any(p.exists() for p in created)
