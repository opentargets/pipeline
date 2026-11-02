"""Download a file once per staging date and record what was fetched."""

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime

import requests
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle
from otter.util.errors import NotFoundError, OtterError

SIDECAR_SUFFIX = '.fetch.json'


class FetchError(OtterError):
    """A download failed or returned nothing."""


@dataclass(frozen=True)
class FetchRecord:
    """What a download fetched, as written to the sidecar file."""

    source_url: str
    fetched_at: str
    size: int
    sha256: str
    extra: dict[str, str]


def _now() -> datetime:
    return datetime.now(UTC)


def _exists(handle: StorageHandle) -> bool:
    try:
        handle.stat()
    except NotFoundError:
        return False
    return True


def fetch_once(
    url: str,
    destination: str,
    config: Config,
    as_of: str,
    extra: dict[str, str] | None = None,
    timeout: int = 300,
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

    Returns:
        None when the destination already existed and was kept, the record of
        the download otherwise.

    Raises:
        FetchError: If the response is not 2xx or its body is empty.
    """
    target = StorageHandle(destination, config=config)
    sidecar = StorageHandle(destination + SIDECAR_SUFFIX, config=config)

    if _exists(target):
        logger.info(f'keeping {destination}, already fetched')
        if not _exists(sidecar):
            logger.warning(f'no {destination + SIDECAR_SUFFIX} for {destination}; its provenance is unknown')
        return None

    logger.info(f'fetching {url} to {destination}')
    response = requests.get(url, timeout=timeout)
    if not response.ok:
        raise FetchError(f'{url} returned status {response.status_code}')
    if not response.content:
        raise FetchError(f'{url} returned an empty body with status {response.status_code}')

    record = FetchRecord(
        source_url=response.url,
        fetched_at=_now().strftime('%Y-%m-%dT%H:%M:%SZ'),
        size=len(response.content),
        sha256=hashlib.sha256(response.content).hexdigest(),
        extra=dict(extra or {}),
    )
    target.write(response.content)
    sidecar.write_text(json.dumps(asdict(record), sort_keys=True, indent=2))

    if record.fetched_at[:10] != as_of:
        logger.warning(f'{destination} was fetched on {record.fetched_at[:10]}, not on the staging date {as_of}')
    return record
