"""Download a file once per staging date and record what was fetched."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import requests
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError, OtterError, TaskValidationError

SIDECAR_SUFFIX = '.fetch.json'


class FetchError(OtterError):
    """A download failed or returned nothing."""


@dataclass(frozen=True)
class FetchRecord:
    """What a download fetched, as written to the sidecar file."""

    source_url: str
    requested_url: str
    fetched_at: str
    size: int
    sha256: str
    content_type: str | None
    last_modified: str | None
    etag: str | None
    extra: dict[str, str]


def _now() -> datetime:
    return datetime.now(UTC)


def _exists(handle: StorageHandle) -> bool:
    try:
        handle.stat()
    except NotFoundError:
        return False
    return True


def is_fetched(destination: str, config: Config) -> bool:
    """Whether `destination` already exists."""
    return _exists(StorageHandle(destination, config=config))


def keep_existing(destination: str, config: Config) -> None:
    """Log that `destination` is kept, and warn when it has no sidecar.

    Args:
        destination: The path, relative to the release root, that already exists.
        config: The otter configuration used to resolve locations.
    """
    logger.info(f'keeping {destination}, already fetched')
    if not _exists(StorageHandle(destination + SIDECAR_SUFFIX, config=config)):
        logger.warning(f'no {destination + SIDECAR_SUFFIX} for {destination}; its provenance is unknown')


def fetch_once(
    url: str,
    destination: str,
    config: Config,
    as_of: str,
    extra: dict[str, str] | None = None,
    timeout: int = 300,
    expect: bytes | None = None,
) -> FetchRecord | None:
    """Download `url` to `destination` unless the destination already exists.

    Args:
        url: The URL to download.
        destination: The path, relative to the release root, to write to. The
            sidecar is written beside it, with `SIDECAR_SUFFIX` appended.
        config: The otter configuration used to resolve locations.
        as_of: The staging date, `YYYY-MM-DD`. A download on another date logs a
            warning.
        extra: Additional fields to record in the sidecar.
        timeout: The request timeout in seconds.
        expect: The prefix the body must start with, when set.

    Returns:
        None when the destination already existed and was kept, the record of
        the download otherwise.

    Raises:
        FetchError: If the response is not 2xx, its body is empty, its content
            type is HTML, or its body does not start with `expect`.
    """
    target = StorageHandle(destination, config=config)
    sidecar = StorageHandle(destination + SIDECAR_SUFFIX, config=config)

    if _exists(target):
        keep_existing(destination, config)
        return None

    logger.info(f'fetching {url} to {destination}')
    response = requests.get(url, timeout=timeout)
    if not response.ok:
        raise FetchError(f'{url} returned status {response.status_code}')
    if not response.content:
        raise FetchError(f'{url} returned an empty body with status {response.status_code}')
    content_type = response.headers.get('Content-Type')
    if content_type and 'text/html' in content_type.lower():
        raise FetchError(f'{url} returned {content_type}, not a data file')
    if expect is not None and not response.content.startswith(expect):
        raise FetchError(f'{url} does not start with {expect!r}; first bytes: {response.content[:32]!r}')

    record = FetchRecord(
        source_url=response.url,
        requested_url=url,
        fetched_at=_now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        size=len(response.content),
        sha256=hashlib.sha256(response.content).hexdigest(),
        content_type=content_type,
        last_modified=response.headers.get('Last-Modified'),
        etag=response.headers.get('ETag'),
        extra=dict(extra or {}),
    )
    target.write(response.content)
    sidecar.write_text(json.dumps(asdict(record), sort_keys=True, indent=2))

    if record.fetched_at[:10] != as_of:
        logger.warning(f'{destination} was fetched on {record.fetched_at[:10]}, not on the staging date {as_of}')
    return record


def validate_destination(destination: str, config: Config) -> None:
    """Check that a fetched file exists and is not empty.

    Args:
        destination: The path, relative to the release root, that was fetched to.
        config: The otter configuration used to resolve locations.

    Raises:
        TaskValidationError: If the destination does not exist or is empty.
    """
    try:
        size = StorageHandle(destination, config=config).stat().size
    except Exception as e:
        raise TaskValidationError(f'{destination} does not exist') from e
    if not size:
        logger.error(f'{destination} is empty')
        raise TaskValidationError(f'{destination} is empty')
