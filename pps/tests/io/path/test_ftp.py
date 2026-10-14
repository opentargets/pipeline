"""Test ftp module."""

import pytest


class TestFtpPath:
    def test_initialization(self):
        from pps.io.path.ftp import FTPPath

        ftp_path = FTPPath('ftp://example.com/path/to/file.txt')
        assert ftp_path.uri == 'ftp://example.com/path/to/file.txt'
        assert ftp_path.server == 'example.com'
        assert ftp_path.filename == 'file.txt'
        assert ftp_path.base_dir == '/path/to'

    @pytest.mark.parametrize(
        ('uri', 'expected_error'),
        [
            pytest.param('http://example.com/path/to/file.txt', 'Unsupported URL scheme', id='unsupported_scheme'),
            pytest.param('ftp:///path/to/file.txt', 'FTP server is missing', id='missing_server'),
            pytest.param('ftp://example.com/', 'File name is missing', id='missing_file_name'),
        ],
    )
    def test_invalid_scheme(self, uri, expected_error):
        from pps.errors import PpsError
        from pps.io.path.ftp import FTPPath

        with pytest.raises(PpsError, match=expected_error):
            FTPPath(uri)

    def test_repr(self):
        from pps.io.path.ftp import FTPPath

        ftp_path = FTPPath('ftp://example.com/path/to/file.txt')
        assert repr(ftp_path) == 'ftp://example.com/path/to/file.txt'
        assert str(ftp_path) == 'ftp://example.com/path/to/file.txt'
