"""Split a ClickHouse SQL script into statements."""

QUOTES = "'\"`"


def split_statements(sql: str) -> list[str]:
    """Split on `;` outside quotes, backtick identifiers and comments; drop statements with no code."""
    statements: list[str] = []
    start = 0
    has_code = False
    i, n = 0, len(sql)
    while i < n:
        c = sql[i]
        if c in QUOTES:
            i = _end_of_quoted(sql, i)
            has_code = True
            continue
        if sql.startswith('--', i):
            end = sql.find('\n', i)
            i = n if end == -1 else end + 1
            continue
        if sql.startswith('/*', i):
            end = sql.find('*/', i + 2)
            i = n if end == -1 else end + 2
            continue
        if c == ';':
            if has_code:
                statements.append(sql[start:i].strip())
            start, has_code = i + 1, False
        elif not c.isspace():
            has_code = True
        i += 1
    if has_code:
        statements.append(sql[start:].strip())
    return statements


def _end_of_quoted(sql: str, i: int) -> int:
    """The index after the quoted span opening at `i`; a backslash or a doubled quote escapes."""
    q = sql[i]
    i += 1
    while i < len(sql):
        if sql[i] == '\\':
            i += 2
        elif sql[i] == q:
            if sql.startswith(q, i + 1):
                i += 2
            else:
                return i + 1
        else:
            i += 1
    return i
