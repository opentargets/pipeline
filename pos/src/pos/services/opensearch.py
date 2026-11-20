"""OpenSearch client for the running server."""

from opensearchpy import OpenSearch

from pos.readiness import wait_until_ready

# the query-insights plugin in the stock image writes top_queries-* indices
QUERY_INSIGHTS_OFF = {
    'search.insights.top_queries.latency.enabled': False,
    'search.insights.top_queries.cpu.enabled': False,
    'search.insights.top_queries.memory.enabled': False,
}
TOP_QUERIES_PATTERN = 'top_queries-*'


def connect(host: str = 'localhost', port: int = 9200) -> OpenSearch:
    """A client for the server at host:port, once the cluster is at least yellow."""
    wait_until_ready(f'http://{host}:{port}/_cluster/health?wait_for_status=yellow&timeout=5s')
    return OpenSearch([{'host': host, 'port': port}], use_ssl=False, timeout=7200)


def disable_query_insights(client: OpenSearch) -> None:
    """Turn off top-N query collection, so no top_queries-* index is written."""
    client.cluster.put_settings(body={'persistent': QUERY_INSIGHTS_OFF})


def top_queries_indices(client: OpenSearch) -> list[str]:
    """The query-insights indices on the cluster, hidden ones included."""
    found = client.indices.get(index=TOP_QUERIES_PATTERN, expand_wildcards='all', ignore_unavailable=True)
    return sorted(found)
