"""Test GCS path module."""

import pytest


class TestGCSPath:
    def test_initialization(self):
        from pps.io.path.gcs import GCSPath

        gcs_path = GCSPath('gs://bucket/path/to/file.txt')
        assert gcs_path.uri == 'gs://bucket/path/to/file.txt'
        assert gcs_path.bucket == 'bucket'
        assert gcs_path.object == 'path/to/file.txt'

    def test_repr(self):
        from pps.io.path.gcs import GCSPath

        gcs_path = GCSPath('gs://bucket/path/to/file.txt')
        assert repr(gcs_path) == 'gs://bucket/path/to/file.txt'
        assert str(gcs_path) == 'gs://bucket/path/to/file.txt'

    @pytest.mark.parametrize(
        ('uri', 'expected_error'),
        [
            pytest.param('http://bucket/path/to/file.txt', 'Unsupported URL scheme', id='unsupported_scheme'),
            pytest.param('gs://', 'Bucket name is missing', id='missing_bucket'),
            pytest.param('gs://bucket/', 'File name is missing', id='missing_file_name'),
        ],
    )
    def test_invalid_scheme(self, uri, expected_error):
        from pps.errors import PpsError
        from pps.io.path.gcs import GCSPath

        with pytest.raises(PpsError, match=expected_error):
            GCSPath(uri)
