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


def _staged(bucket: MagicMock, size: int, *, verified: int | None = None) -> None:
    """Put an object of ``size`` bytes at the destination.

    ``verified`` records a size this operator checked against the source, as a
    real staged object would carry in its metadata.
    """
    existing = MagicMock()
    existing.size = size
    existing.metadata = None if verified is None else {'staged_verified_bytes': str(verified)}
    bucket.get_blob.return_value = existing


def _download(
    http: MagicMock, body: bytes, content_length: int | None, encoding: str | None = None
) -> None:
    r = MagicMock()
    r.__enter__.return_value = r
    headers = {} if content_length is None else {'Content-Length': str(content_length)}
    if encoding is not None:
        headers['Content-Encoding'] = encoding
    r.headers = headers
    r.iter_content.return_value = [body]
    http.get.return_value = r


def _head(http: MagicMock, content_length: int, encoding: str | None = None) -> None:
    r = MagicMock()
    headers = {'Content-Length': str(content_length)}
    if encoding is not None:
        headers['Content-Encoding'] = encoding
    r.headers = headers
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


def test_keeps_verified_object_when_source_size_is_unknown(gcs, http):
    """An unreachable source is no reason to discard a copy we already verified."""
    _staged(gcs, len(PAYLOAD), verified=len(PAYLOAD))
    http.head.side_effect = requests.RequestException('network down')

    _operator(skip_if_exists=True).execute({})

    http.get.assert_not_called()
    gcs.blob.return_value.upload_from_filename.assert_not_called()


def test_restages_unverified_object_when_source_size_is_unknown(gcs, http):
    """An object this operator never checked must not be trusted on a failed HEAD.

    It also recovers the case where HEAD fails but GET works.
    """
    _staged(gcs, len(PAYLOAD))  # no recorded verification
    http.head.side_effect = requests.RequestException('network down')
    _download(http, PAYLOAD, len(PAYLOAD))

    _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_restages_when_verified_size_no_longer_matches(gcs, http):
    """Recorded metadata that disagrees with the object's size proves nothing."""
    _staged(gcs, 4242, verified=len(PAYLOAD))
    http.head.side_effect = requests.RequestException('network down')
    _download(http, PAYLOAD, len(PAYLOAD))

    _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_verified_size_is_recorded_on_the_object(gcs, http):
    """The checked size is stored so a later run can tell verified from found."""
    _download(http, PAYLOAD, len(PAYLOAD))

    _operator(skip_if_exists=True).execute({})

    assert gcs.blob.return_value.metadata == {'staged_verified_bytes': str(len(PAYLOAD))}


def test_unverified_download_is_not_marked_as_verified(gcs, http):
    """No Content-Length means nothing was checked; do not claim otherwise."""
    _download(http, PAYLOAD, None)

    _operator(skip_if_exists=True).execute({})

    blob = gcs.blob.return_value
    blob.upload_from_filename.assert_called_once()
    assert 'staged_verified_bytes' not in (blob.metadata or {})


def test_encoded_response_is_not_compared_against_decoded_bytes(gcs, http):
    """Content-Length counts encoded bytes; iter_content yields decoded ones.

    Comparing the two would reject every attempt and block staging entirely.
    """
    _download(http, PAYLOAD, 12345, encoding='gzip')

    _operator(skip_if_exists=True).execute({})

    gcs.blob.return_value.upload_from_filename.assert_called_once()


def test_encoded_head_does_not_trigger_a_pointless_restage(gcs, http):
    """The same guard applies to the size the skip check reads."""
    _staged(gcs, len(PAYLOAD), verified=len(PAYLOAD))
    _head(http, 12345, encoding='gzip')

    _operator(skip_if_exists=True).execute({})

    http.get.assert_not_called()
    gcs.blob.return_value.upload_from_filename.assert_not_called()


def test_requests_ask_for_identity_encoding(gcs, http):
    """Prefer not to receive an encoded body in the first place."""
    _download(http, PAYLOAD, len(PAYLOAD))
    _head(http, len(PAYLOAD))
    _staged(gcs, 1)

    _operator(skip_if_exists=True).execute({})

    assert http.head.call_args.kwargs['headers'] == {'Accept-Encoding': 'identity'}
    assert http.get.call_args.kwargs['headers'] == {'Accept-Encoding': 'identity'}


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
