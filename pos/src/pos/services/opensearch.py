"""OpenSearch client for the running server."""

from opensearchpy import OpenSearch

from pos.readiness import wait_until_ready


def connect(host: str = 'localhost', port: int = 9200) -> OpenSearch:
    """A client for the server at host:port, once the cluster is at least yellow."""
    wait_until_ready(f'http://{host}:{port}/_cluster/health?wait_for_status=yellow&timeout=5s')
    return OpenSearch([{'host': host, 'port': port}], use_ssl=False, timeout=7200)
