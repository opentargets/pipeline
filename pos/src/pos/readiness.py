"""Wait for a server's HTTP endpoint to answer."""

import http.client
import time
import urllib.error
import urllib.request

from loguru import logger

REQUEST_TIMEOUT_S = 30.0


def wait_until_ready(url: str, *, timeout_s: float = 600, interval_s: float = 5, expect: str | None = None) -> None:
    """Poll `url` until it returns HTTP 200 (and, with `expect`, that body), else raise TimeoutError."""
    deadline = time.monotonic() + timeout_s
    while True:
        remaining = deadline - time.monotonic()
        try:
            with urllib.request.urlopen(url, timeout=max(min(REQUEST_TIMEOUT_S, remaining), 0.1)) as r:  # noqa: S310
                body = r.read().decode(errors='replace').strip()
                if r.status == 200 and (expect is None or body == expect):
                    logger.debug(f'{url} is ready')
                    return
                last = f'HTTP {r.status}: {body[:200]!r}'
        except urllib.error.HTTPError as e:
            last = f'HTTP {e.code}: {e.reason}'
        except (OSError, http.client.HTTPException) as e:
            last = repr(e)
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError(f'{url} not ready after {timeout_s}s: {last}')
        logger.debug(f'{url} not ready ({last}), {remaining:.0f}s left')
        time.sleep(min(interval_s, remaining))
