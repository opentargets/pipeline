"""wait_until_ready polls an HTTP endpoint until it answers."""

import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from pos.readiness import wait_until_ready


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    class Handler(BaseHTTPRequestHandler):
        status = 200
        body = b'Ok.\n'

        def do_GET(self):
            self.send_response(Handler.status)
            self.end_headers()
            self.wfile.write(Handler.body)

        def log_message(self, format, *args):
            pass

    httpd = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield httpd, Handler
    httpd.shutdown()
    httpd.server_close()


def test_wait_until_ready_times_out():
    start = time.monotonic()
    with pytest.raises(TimeoutError):
        wait_until_ready(f'http://127.0.0.1:{_free_port()}/ping', timeout_s=0.2, interval_s=0.05)
    assert time.monotonic() - start < 5


def test_wait_until_ready_ok(server):
    httpd, _ = server
    start = time.monotonic()
    wait_until_ready(f'http://127.0.0.1:{httpd.server_port}/ping', timeout_s=5, interval_s=5, expect='Ok.')
    assert time.monotonic() - start < 1


def test_wait_until_ready_rejects_error_status(server):
    httpd, handler = server
    handler.status = 408
    with pytest.raises(TimeoutError, match='408'):
        wait_until_ready(f'http://127.0.0.1:{httpd.server_port}/_cluster/health', timeout_s=0.2, interval_s=0.05)


def test_wait_until_ready_rejects_unexpected_body(server):
    httpd, handler = server
    handler.body = b'starting'
    with pytest.raises(TimeoutError):
        wait_until_ready(f'http://127.0.0.1:{httpd.server_port}/ping', timeout_s=0.2, interval_s=0.05, expect='Ok.')
