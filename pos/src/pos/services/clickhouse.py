"""ClickHouse client for the running server."""

import clickhouse_connect
from clickhouse_connect.driver.client import Client

from pos.readiness import wait_until_ready


def connect(
    *,
    host: str = 'localhost',
    port: int = 8123,
    username: str = 'default',
    password: str = '',
    database: str = 'default',
) -> Client:
    """A client for the server at host:port, once its /ping answers `Ok.`."""
    wait_until_ready(f'http://{host}:{port}/ping', expect='Ok.')
    return clickhouse_connect.get_client(
        host=host,
        port=port,
        username=username,
        password=password,
        database=database,
        send_receive_timeout=1800,
    )


def create_database(client: Client, database: str, exists_ok: bool = True) -> None:
    """Create a database, with IF NOT EXISTS when `exists_ok`."""
    if exists_ok:
        query = 'CREATE DATABASE IF NOT EXISTS {database:Identifier}'
    else:
        query = 'CREATE DATABASE {database:Identifier}'
    client.query(query=query, parameters={'database': database})


def tables(client: Client, database: str) -> list[str]:
    """The tables in `database`."""
    query = 'SELECT name FROM system.tables WHERE database = {database:String} ORDER BY name'
    return [row[0] for row in client.query(query=query, parameters={'database': database}).result_rows]
