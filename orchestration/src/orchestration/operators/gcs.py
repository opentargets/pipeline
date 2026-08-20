"""Custom operators for Google Cloud Storage (GCS) interactions."""

import tempfile
from collections.abc import Sequence
from pathlib import Path

import requests
from airflow.providers.google.cloud.hooks.gcs import GCSHook
from airflow.sdk import BaseOperator
from google.cloud.exceptions import NotFound
from google.cloud.storage import Client
from google.cloud.storage.bucket import Bucket

from orchestration.utils import GCSPath
from orchestration.utils.common import GCP_PROJECT_PLATFORM


class UploadFileOperator(BaseOperator):
    """Custom operator that uploads a file to GCS.

    This operator will create a GCS bucket if it does not exist and upload the
    file to the specified path inside that bucket.

    Args:
        project_id: The GCP project ID. Defaults to the platform project.
        src_path: The path to the file to upload.
        dst_uri: The destination URI in GCS.
    """

    template_fields: Sequence[str] = ('src_path', 'dst_uri')

    def __init__(
        self,
        *args,
        project_id: str = GCP_PROJECT_PLATFORM,
        src_path: Path,
        dst_uri: str,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.project_id = project_id
        self.dst_uri = GCSPath(dst_uri)
        self.src_path = src_path

        self.bucket_name, self.path = self.dst_uri.split()

    def execute(self, context) -> None:
        """Execute the Operator."""
        c = Client(project=self.project_id)
        b = Bucket(client=c, name=self.bucket_name)

        if not b.exists():
            b.create()

        blob = b.blob(self.path)
        blob.upload_from_filename(self.src_path)
        self.log.info('uploaded file from %s to: %s', self.src_path, self.dst_uri)


class UploadRemoteFileOperator(BaseOperator):
    """Custom operator that uploads a remote file to GCS.

    This operator will create a GCS bucket if it does not exist and upload the
    file from a URL to a path inside that bucket.

    Args:
        project_id: The GCP project ID. Defaults to the platform project.
        src_url: Source file URL.
        dst_uri: The destination URI in GCS.
        skip_if_exists: Skip the transfer when the destination already holds the
            source's bytes. The object name alone is not taken as proof: the size
            is checked against the source, so a truncated or hand-staged object
            under the expected name is re-staged rather than trusted.
        timeout: Per-read/connect timeout, in seconds, for the source request.
    """

    template_fields: Sequence[str] = ('src_url', 'dst_uri')

    def __init__(
        self,
        *args,
        project_id: str = GCP_PROJECT_PLATFORM,
        src_url: str,
        dst_uri: str,
        skip_if_exists: bool = False,
        timeout: int = 60,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.project_id = project_id
        self.src_url = src_url
        self.dst_uri = GCSPath(dst_uri)
        self.skip_if_exists = skip_if_exists
        self.timeout = timeout

        self.bucket_name, self.path = self.dst_uri.split()

    def _source_size(self) -> int | None:
        """Content-Length the source advertises, or None when unavailable."""
        try:
            r = requests.head(self.src_url, timeout=self.timeout, allow_redirects=True)
            r.raise_for_status()
            size = r.headers.get('Content-Length')
            return int(size) if size is not None else None
        except (requests.RequestException, ValueError) as err:
            self.log.warning('could not determine the size of %s: %s', self.src_url, err)
            return None

    def _can_skip(self, bucket: Bucket) -> bool:
        """Whether the destination already holds the source's bytes.

        A version-pinned artifact staged by an earlier run is the common case, but
        an object can also be short (an interrupted stream) or simply the wrong
        file staged by hand under the right name. Both would otherwise be trusted
        forever, so the existing object is accepted only when its size matches the
        source. When the source size cannot be read the existing object is kept —
        an unreachable source is no reason to discard a good local copy.
        """
        if not (self.skip_if_exists and bucket.exists()):
            return False

        blob = bucket.get_blob(self.path)
        if blob is None:
            return False

        expected = self._source_size()
        if expected is None:
            self.log.info('destination %s exists, source size unknown, keeping it', self.dst_uri)
            return True
        if blob.size == expected:
            self.log.info('destination %s already holds %s bytes, skipping', self.dst_uri, expected)
            return True

        self.log.warning(
            'destination %s is %s bytes but the source is %s bytes, re-staging',
            self.dst_uri,
            blob.size,
            expected,
        )
        return False

    def execute(self, context) -> None:
        c = Client(project=self.project_id)
        b = Bucket(client=c, name=self.bucket_name)

        if self._can_skip(b):
            return

        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            temp_file = Path(tmp_file.name)

        try:
            with requests.get(self.src_url, stream=True, timeout=self.timeout) as r:
                r.raise_for_status()
                expected = r.headers.get('Content-Length')
                with open(temp_file, 'wb') as f:
                    f.writelines(r.iter_content(chunk_size=8192))

            # A stream that ends early can still look like a clean download, and
            # uploading it would publish a corrupt artifact that skip_if_exists
            # then has to catch on every later run. Refuse it here instead.
            downloaded = temp_file.stat().st_size
            if expected is not None and downloaded != int(expected):
                raise ValueError(
                    f'{self.src_url} returned {downloaded} bytes, expected {expected}; '
                    'refusing to upload a truncated file.'
                )

            if not b.exists():
                b.create()

            blob = b.blob(self.path)
            blob.upload_from_filename(temp_file)
            self.log.info(
                'uploaded %s bytes from %s to: %s', downloaded, self.src_url, self.dst_uri
            )
        finally:
            # A ~629 MB artifact left behind on every failed attempt fills the
            # worker's disk across retries.
            temp_file.unlink(missing_ok=True)


class UploadStringOperator(BaseOperator):
    """Custom operator that uploads a string to GCS.

    This operator will create a GCS bucket if it does not exist and upload the
    given string to the specified path inside that bucket.

    An error will be raised if a file with the destination name already exists
    and overwrite is not set to `True`.

    Args:
        project_id: The GCP project ID. Defaults to the platform project.
        contents: The string to upload.
        dst_uri: The destination URI in GCS.
    """

    template_fields: Sequence[str] = ('dst_uri',)

    def __init__(
        self,
        *args,
        project_id: str = GCP_PROJECT_PLATFORM,
        contents: str,
        dst_uri: str,
        overwrite: bool = False,
        gcp_conn_id: str = 'google_cloud_default',
        impersonation_chain: str | Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.project_id = project_id
        self.dst_uri = GCSPath(dst_uri)
        self.contents = contents
        self.overwrite = overwrite
        self.gcp_conn_id = gcp_conn_id
        self.impersonation_chain = impersonation_chain
        self.bucket_name, self.path = self.dst_uri.split()

    def execute(self, context) -> None:
        """Execute the Operator."""
        hook = GCSHook(
            gcp_conn_id=self.gcp_conn_id,
            impersonation_chain=self.impersonation_chain,
        )

        c = hook.get_conn()
        try:
            c.get_bucket(self.bucket_name)
        except NotFound:
            hook.create_bucket(bucket_name=self.bucket_name)

        if not self.overwrite and hook.exists(self.bucket_name, self.path):
            raise FileExistsError(f'Destination object {self.dst_uri} already exists.')

        hook.upload(
            bucket_name=self.bucket_name,
            object_name=self.path,
            data=self.contents,
        )
        self.log.info('uploaded string to: %s', self.dst_uri)


class CopyBlobOperator(BaseOperator):
    """Custom operator that copies a GCS blob to another location.

    The operator will make sure the source blob exists and will raise an error
    otherwise. Regarding the destination file, an error will be raised if it
    already exists and overwrite is not set to `True`.

    Args:
        src_uri: The source GCS URI.
        dst_uri: The destination GCS URI.
        overwrite: Whether to overwrite the destination file if it already exists.
        gcp_conn_id: The connection ID to use when connecting to GCS.
        impersonation_chain: The service account to impersonate.
    """

    template_fields: Sequence[str] = ('src_uri', 'dst_uri', 'impersonation_chain')

    def __init__(
        self,
        *,
        src_uri: str,
        dst_uri: str,
        overwrite: bool = False,
        gcp_conn_id: str = 'google_cloud_default',
        impersonation_chain: str | Sequence[str] | None = None,
        **kwargs,
    ) -> None:
        self.src_uri = src_uri
        self.dst_uri = dst_uri
        self.overwrite = overwrite
        self.gcp_conn_id = gcp_conn_id
        self.impersonation_chain = impersonation_chain

        super().__init__(**kwargs)

    def execute(self, context) -> None:
        """Execute the Operator."""
        hook = GCSHook(
            gcp_conn_id=self.gcp_conn_id,
            impersonation_chain=self.impersonation_chain,
        )

        source_bucket, source_object = self.src_uri.removeprefix('gs://').split('/', 1)
        destination_bucket, destination_object = self.dst_uri.removeprefix('gs://').split('/', 1)

        if not hook.exists(source_bucket, source_object):
            raise FileNotFoundError(f'Source object {self.src_uri} does not exist.')
        if not self.overwrite and hook.exists(destination_bucket, destination_object):
            raise FileExistsError(f'Destination object {self.dst_uri} already exists.')

        self.log.info('copying %s to %s', self.src_uri, self.dst_uri)
        hook.rewrite(source_bucket, source_object, destination_bucket, destination_object)
