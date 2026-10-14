"""pps exceptions module."""

from enum import Enum


class PpsErrorMessage(Enum):
    """Base class for all exceptions in the pps package."""

    UNSUPPORTED_URL_SCHEME = 'Unsupported URL scheme: {scheme}'
    BUCKET_NAME_MISSING = 'Bucket name is missing in the URL: {url}'
    FILE_NAME_MISSING = 'File name is missing in the URL: {url}'
    GCS_CLIENT_INITIALIZATION_FAILED = 'Failed to initialize Google Cloud Storage client: {error}'
    FTP_SERVER_MISSING = 'FTP server is missing in the URL: {url}'
    INVALID_TRANSFERABLE_OBJECTS = (
        'Invalid transferable objects provided. Expected FTPtoGCPTransferableObject instances.'
    )
    DOWNLOAD_STUDIES_EMPTY = 'List of downloaded studies from GWAS Catalog release is empty: {path}'
    PREVIOUS_CURATION_EMPTY = 'Previous curation data is empty: {path}'
    FAILED_TO_FETCH = 'Failed to fetch the release information from the {uri}'
    MISSING_RELEASE_DATE_TEMPLATE = (
        'The destination must contain a template for the release date, e.g. some/path/{release_date}/file.txt.'
    )
    EMPTY_TRANSFERABLE_OBJECTS = 'Transferable objects list cannot be empty.'


class PpsError(Exception):
    """Base class for the pps exceptions."""

    def __init__(self, message: PpsErrorMessage, **kwargs: str) -> None:
        """Initialize the PpsError exception.

        Args:
            message (PpsErrorMessage): The error message.
            **kwargs (str): Additional arguments to format the message.
        """
        super().__init__(message.value.format(**kwargs))


__all__ = ['PpsError', 'PpsErrorMessage']
