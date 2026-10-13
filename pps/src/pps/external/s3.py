"""Module for S3 configuration."""

from __future__ import annotations

import os
from typing import ClassVar
from urllib.parse import urlsplit

from pydantic import SecretStr

from pps.external import ExternalConfig

DEFAULT_ENDPOINT_URL = 'https://s3.amazonaws.com'
"""AWS S3 endpoint used when `AWS_ENDPOINT_URL` is not set."""
HTTPS_PORT = 443
"""Port of an endpoint that sets none: the S3 connector always connects over https."""


def parse_endpoint_url(endpoint_url: str) -> tuple[str, int]:
    """Split an S3 endpoint URL into its host and port.

    The endpoint is either `host[:port]` or `https://host[:port]`, without a path. The
    port defaults to 443, as the S3 connector always connects over https.

    Args:
        endpoint_url (str): The endpoint URL, as in the `AWS_ENDPOINT_URL` environment variable.

    Returns:
        tuple[str, int]: The host and port of the endpoint.

    Raises:
        ValueError: If the endpoint is not a valid `host[:port]` or `https://host[:port]`.
    """
    invalid = f"AWS_ENDPOINT_URL '{endpoint_url}' must be 'host[:port]' or 'https://host[:port]'"
    url = urlsplit(endpoint_url if '://' in endpoint_url else f'https://{endpoint_url}')
    if url.scheme != 'https':
        raise ValueError(f'{invalid}: the S3 connector only connects over https.')
    if url.path not in ('', '/') or url.query or url.fragment or url.username or not url.hostname:
        raise ValueError(f'{invalid}.')
    try:
        port = url.port
    except ValueError as e:
        raise ValueError(f'{invalid}: the port must be an integer.') from e
    return url.hostname, port or HTTPS_PORT


class S3Config(ExternalConfig):
    """Model for S3 configuration.

    Note:
        This configuration is used to connect to S3 compatible storage.
        Ensure that the access key and secret key have the necessary permissions
        to access the specified bucket. This configuration can be loaded from a JSON file
        using the `from_json` class method.

    Examples:
    ---
    >>> config = S3Config(
    ...     bucket_name="my-bucket",
    ...     s3_host_port=9000,
    ...     s3_host_url="s3.my-domain.com",
    ...     access_key_id="my-access-key-id",
    ...     secret_access_key="my-secret-access-key",
    ... )
    >>> print(config.bucket_name)
    my-bucket
    >>> print(config.s3_host_port)
    9000
    >>> print(config.s3_host_url)
    s3.my-domain.com
    >>> print(config.access_key_id.get_secret_value())
    my-access-key-id
    >>> print(config.secret_access_key.get_secret_value())
    my-secret-access-key
    """

    _HADOOP_CONNECTOR_PKG: ClassVar[str] = (
        'org.apache.hadoop:hadoop-aws:3.3.6,com.amazonaws:aws-java-sdk-bundle:1.12.367'
    )
    """Connector for AWS S3 compatible storage.
        See https://mvnrepository.com/artifact/org.apache.hadoop/hadoop-aws/3.3.6"""

    bucket_name: str
    """Name of the bucket, without s3:// or s3a:// prefix."""
    s3_host_port: int
    """Port number of the S3 host."""
    s3_host_url: str
    """URL of the S3 host."""
    access_key_id: SecretStr
    """Access key ID for S3 authentication."""
    secret_access_key: SecretStr
    """Secret access key for S3 authentication."""

    @classmethod
    def from_env(cls) -> S3Config:
        """Load S3 configuration from environment variables.

        Returns:
            S3Config: S3 configuration instance.

        !!! note:
            The following environment variables must be set:
            - `AWS_S3_BUCKET_NAME`: Name of the S3 bucket.
            - `AWS_ENDPOINT_URL`: Endpoint URL of the S3 service, as `host[:port]` or
              `https://host[:port]`. The port defaults to 443. Defaults to `https://s3.amazonaws.com`.
            - `AWS_ACCESS_KEY_ID`: Access key ID for S3 authentication.
            - `AWS_SECRET_ACCESS_KEY`: Secret access key for S3 authentication.

        Examples:
        ---
        >>> import os
        >>> os.environ["AWS_S3_BUCKET_NAME"] = "my-bucket"
        >>> os.environ["AWS_ENDPOINT_URL"] = "s3.my-domain.com:9000"
        >>> os.environ["AWS_ACCESS_KEY_ID"] = "my-access-key-id"
        >>> os.environ["AWS_SECRET_ACCESS_KEY"] = "my-secret-access-key"
        >>> config = S3Config.from_env()
        >>> config.bucket_name
        'my-bucket'
        """
        bucket = os.getenv('AWS_S3_BUCKET_NAME')
        if bucket is None:
            raise ValueError('AWS_S3_BUCKET_NAME environment variable is not set.')

        endpoint_url = os.getenv('AWS_ENDPOINT_URL', DEFAULT_ENDPOINT_URL)
        if not endpoint_url:
            raise ValueError('AWS_ENDPOINT_URL environment variable is not set.')

        s3_host_url, s3_host_port = parse_endpoint_url(endpoint_url)

        access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
        if access_key_id is None:
            raise ValueError('AWS_ACCESS_KEY_ID environment variable is not set.')

        secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
        if secret_access_key is None:
            raise ValueError('AWS_SECRET_ACCESS_KEY environment variable is not set.')

        return cls(
            bucket_name=bucket,
            s3_host_port=s3_host_port,
            s3_host_url=s3_host_url,
            access_key_id=SecretStr(access_key_id),
            secret_access_key=SecretStr(secret_access_key),
        )

    @classmethod
    def read(cls, path: str | None = None) -> S3Config:
        """Read from file or environment variables.

        Args:
            path (str | None): Optional path to the configuration file. If not provided,
                the method will attempt to load from environment variables.

        Returns:
            S3Config: S3 configuration instance.
        """
        if path is not None:
            return cls.from_json(path)
        return cls.from_env()
