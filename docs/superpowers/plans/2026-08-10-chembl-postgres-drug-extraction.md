# ChEMBL PostgreSQL Drug Extraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce `chembl_molecule`, `chembl_mechanism`, `chembl_target` and `chembl_drug_warning` as parquet from the ChEMBL PostgreSQL dump already restored by `postgres_export`, retiring the five ChEMBL Elasticsearch tasks.

**Architecture:** `PostgresExportSpec` gains a `queries:` list beside `tables:`. Each entry names a SQL file in `pis/src/pis/sql/` and declares the tables `pg_restore` must load. One restore serves both table exports and query exports. The SQL runs in DuckDB against the attached PostgreSQL server, so nested `STRUCT`/`LIST` shapes serialise straight to parquet.

**Tech Stack:** Python 3.11, Pydantic v2, DuckDB (attached `postgres` scanner), `pixeltable-pgserver` (embedded PostgreSQL 18), pytest, uv.

**Spec:** `docs/superpowers/specs/2026-08-10-chembl-postgres-drug-extraction-design.md`

## Global Constraints

- Branch `chembl-postgres-drug`, stacked on `il-4458`. Do not rebase onto `main`.
- Python `>=3.11,<3.14`. All commands use `uv run --frozen`; the lockfile is authoritative.
- Lint with `ruff check` and type-check with `ty check`; both must pass before every commit.
- Single quotes for inline strings, double for multiline/docstrings. Line length 120. Google-convention docstrings.
- Commit messages must start with the package name: `pis: ...` or `pts: ...`.
- Never add `Co-Authored-By` lines or any Claude/Anthropic/Opus reference to commit messages.
- All work happens in the `pis` package unless a task says otherwise. Run package commands with `--directory pis` or from inside `pis/`.
- The four blocking defects from the PR #17 review should land on `il-4458` before Task 9. They are not part of this plan.

---

## File Structure

| File | Responsibility |
| --- | --- |
| `pis/src/pis/tasks/postgres_export.py` | Modified. Adds `QuerySpec`, `queries` field, SQL loading, restore-list union, query export loop. |
| `pis/src/pis/sql/__init__.py` | Created. Makes `pis.sql` an importable package so `importlib.resources` can read the SQL. |
| `pis/tests/sql/__init__.py`, `pis/tests/sql/_test_demo.sql` | Created. Test-only query fixture, deliberately outside `src/` so it never ships. |
| `pis/src/pis/sql/chembl_drug_warning.sql` | Created. Rebuilds the drug warning document. |
| `pis/src/pis/sql/chembl_mechanism.sql` | Created. Rebuilds the mechanism document. |
| `pis/src/pis/sql/chembl_molecule.sql` | Created. Rebuilds the molecule document. |
| `pis/src/pis/sql/chembl_target.sql` | Created. Rebuilds the target document, including the recursive protein-class walk. |
| `pis/tests/test_postgres_export.py` | Modified. Tests for the spec, the restore union, and query export. |
| `pis/tests/test_chembl_queries.py` | Created. Fixture-database tests for the four SQL files. |
| `/tmp/chembl-baseline/compare_chembl_es.py` | Created outside the repo, never committed. Compares new parquet against the 26.06 JSONL baselines. |
| `pis/config.yaml` | Modified. Adds `queries:`, removes five `elasticsearch` tasks. |
| `pts/config.yaml` | Modified. Repoints four sources to parquet. |
| `pts/src/pts/pyspark/{drug_warning,drug_mechanism_of_action,chembl_molecule,target}.py` | Modified. Format change only. |

The comparison harness is a development tool, not a deliverable. It lives outside the repository and is never committed, so nothing about it can reach the release or the wheel. Its output is the deliverable, pasted into the PR description.

---

### Task 1: `QuerySpec` model and spec validation

**Files:**
- Modify: `pis/src/pis/tasks/postgres_export.py:80-120`
- Create: `pis/src/pis/sql/__init__.py`
- Create: `pis/tests/__init__.py`, `pis/tests/sql/__init__.py`, `pis/tests/sql/_test_demo.sql`
- Test: `pis/tests/test_postgres_export.py`

**Interfaces:**
- Consumes: `PostgresExportSpec`, `TableSpec`, `PostgresExportError` (existing).
- Produces: `QuerySpec` with fields `query: str`, `destination: str`, `requires_tables: list[str]`. `PostgresExportSpec.queries: list[QuerySpec]`. Module constant `QUERY_PACKAGE = 'pis.sql'` and function `_load_query(name: str) -> str`.

- [ ] **Step 1: Create the SQL package**

Create `pis/src/pis/sql/__init__.py` with exactly this content:

```python
"""SQL files run by the postgres_export task's ``queries`` field."""
```

- [ ] **Step 2: Write the failing tests**

Append to `pis/tests/test_postgres_export.py`:

```python
class TestQuerySpec:
    def test_accepts_a_query(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # _test_demo is the only SQL file that exists at this point in the plan;
        # the four real queries arrive in Tasks 5-8
        monkeypatch.setattr(postgres_export, 'QUERY_PACKAGE', 'tests.sql')
        spec = PostgresExportSpec(
            name='postgres_export chembl',
            source='d.tar.gz',
            queries=[
                QuerySpec(
                    query='_test_demo',
                    destination='input/drug/demo.parquet',
                    requires_tables=['molecule_dictionary'],
                )
            ],
        )
        assert spec.tables == []
        assert spec.queries[0].query == '_test_demo'

    def test_rejects_a_spec_with_neither_tables_nor_queries(self) -> None:
        with pytest.raises(ValidationError, match='at least one of tables or queries'):
            PostgresExportSpec(name='postgres_export nothing', source='d.dmp')

    def test_rejects_a_query_with_no_required_tables(self) -> None:
        with pytest.raises(ValidationError):
            QuerySpec(query='chembl_molecule', destination='d.parquet', requires_tables=[])

    def test_rejects_a_query_with_no_sql_file(self) -> None:
        with pytest.raises(ValidationError, match='no SQL file for query'):
            PostgresExportSpec(
                name='postgres_export missing',
                source='d.dmp',
                queries=[QuerySpec(query='does_not_exist', destination='d.parquet', requires_tables=['t'])],
            )

    def test_tables_only_still_works(self) -> None:
        spec = PostgresExportSpec(name='postgres_export some tables', source='d.dmp', tables=[STUDIES])
        assert spec.queries == []


class TestLoadQuery:
    def test_reads_a_sql_file(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # the fixture lives in tests/sql so nothing test-only ships in the wheel
        monkeypatch.setattr(postgres_export, 'QUERY_PACKAGE', 'tests.sql')
        assert 'SELECT' in _load_query('_test_demo').upper()

    def test_raises_for_a_missing_file(self) -> None:
        with pytest.raises(PostgresExportError, match='no SQL file for query'):
            _load_query('does_not_exist')
```

Add `from pis.tasks import postgres_export` to the test module's imports — `monkeypatch.setattr` needs the module object, not the names imported from it.

Extend the existing import block from `pis.tasks.postgres_export` to add `QuerySpec` and `_load_query`.

- [ ] **Step 3: Run the tests to verify they fail**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -k "QuerySpec or LoadQuery" -rxs`
Expected: FAIL with `ImportError: cannot import name 'QuerySpec'`

- [ ] **Step 4: Add the model and the loader**

In `pis/src/pis/tasks/postgres_export.py`, add to the imports:

```python
from importlib import resources
from pydantic import BaseModel, Field, field_validator, model_validator
```

`Self` is already imported from `typing` at the top of the module; do not add it again.

Add the constant next to `DATABASE`:

```python
QUERY_PACKAGE = 'pis.sql'
"""Package holding the SQL files named by ``QuerySpec.query``."""
```

Add after `TableSpec`:

```python
class QuerySpec(BaseModel):
    """One SQL query to run against the restored database and export."""

    query: str
    """Name of the SQL file in :py:obj:`QUERY_PACKAGE`, without the ``.sql`` suffix."""
    destination: str
    """Path for the parquet file, relative to the release root."""
    requires_tables: Annotated[list[str], Field(min_length=1)]
    """Tables ``pg_restore`` must load for this query to run.

    The restore is selective, so a table a query reads but does not declare will
    simply not be there. ChEMBL's large tables make restoring everything
    impractical, which is why this is explicit rather than inferred.
    """
```

Change `PostgresExportSpec.tables` and add `queries` plus the validators:

```python
    tables: list[TableSpec] = []
    """The tables to restore and export verbatim."""
    queries: list[QuerySpec] = []
    """The queries to run against the restored database and export."""

    @field_validator('queries')
    @classmethod
    def _sql_files_exist(cls, value: list[QuerySpec]) -> list[QuerySpec]:
        for query in value:
            # pydantic only converts ValueError and AssertionError into a
            # ValidationError; a PostgresExportError would escape the spec layer
            try:
                _load_query(query.query)
            except PostgresExportError as e:
                raise ValueError(str(e))
        return value

    @model_validator(mode='after')
    def _has_work(self) -> Self:
        if not self.tables and not self.queries:
            raise ValueError('postgres_export needs at least one of tables or queries')
        return self
```

Add the loader beside the other module-level helpers, after `_sql_str`:

```python
def _load_query(name: str) -> str:
    """Read a SQL file shipped inside the package.

    Reading through ``importlib.resources`` rather than a path relative to this
    module means it works the same whether the package is installed in the image
    or run from the source tree.
    """
    try:
        return resources.files(QUERY_PACKAGE).joinpath(f'{name}.sql').read_text(encoding='utf-8')
    except (FileNotFoundError, ModuleNotFoundError, OSError) as e:
        raise PostgresExportError(f'no SQL file for query {name}: {e}')
```

Remove the now-wrong `Annotated[list[TableSpec], Field(min_length=1)]` annotation on `tables`.

- [ ] **Step 5: Create the test-only SQL fixture**

This lives under `tests/`, never under `src/`, so it cannot ship in the wheel or the Docker image.

`importlib.resources` can only read from an importable package, so `pis/tests/` must itself become one. Create `pis/tests/__init__.py` with exactly:

```python
"""Tests for the pis package."""
```

This is verified to work: with it present, `resources.files('tests.sql')` resolves under the pytest rootdir, and the existing suite still collects.

Create `pis/tests/sql/__init__.py` with exactly:

```python
"""SQL fixtures for the postgres_export query tests."""
```

Create `pis/tests/sql/_test_demo.sql` with exactly:

```sql
SELECT txt, count(*) AS n
FROM t
GROUP BY txt
```

Task 3's round-trip test runs this against the demo database. No placeholder file is created — `chembl_drug_warning.sql` is written for the first time in Task 5.

- [ ] **Step 6: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -rxs`
Expected: PASS, all tests including the pre-existing ones.

- [ ] **Step 7: Lint and type-check**

Run: `uv run --frozen --directory pis ruff check src tests`
Run: `uv run --frozen --directory pis ty check src tests`
Expected: `All checks passed!` from both.

- [ ] **Step 8: Commit**

```bash
git add pis/src/pis/sql pis/src/pis/tasks/postgres_export.py pis/tests/test_postgres_export.py
git commit -m "pis: add QuerySpec to postgres_export"
```

---

### Task 2: Restore-list union

**Files:**
- Modify: `pis/src/pis/tasks/postgres_export.py:150-175` (`_build_restore_args`), `:340-355` (`_restore`)
- Test: `pis/tests/test_postgres_export.py`

**Interfaces:**
- Consumes: `QuerySpec` from Task 1.
- Produces: `PostgresExport._restore_table_names(self) -> list[str]`, returning the sorted union of table-export names and every query's `requires_tables`.

- [ ] **Step 1: Write the failing tests**

Append to `pis/tests/test_postgres_export.py`:

```python
class TestRestoreTableNames:
    def _task(self, tables: list[TableSpec], queries: list[QuerySpec], tmp_path: Path) -> PostgresExport:
        spec = PostgresExportSpec(
            name='postgres_export mixed', source='d.dmp', tables=tables, queries=queries
        )
        config = Config(step='demo', steps=['demo'], work_path=tmp_path, pool_size=2, log_level='DEBUG')
        return PostgresExport(spec, TaskContext(config=config, scratchpad=Scratchpad()))

    def test_unions_tables_and_query_requirements(self, tmp_path: Path) -> None:
        task = self._task(
            [STUDIES],
            [
                QuerySpec(
                    query='chembl_drug_warning',
                    destination='d.parquet',
                    requires_tables=['drug_warning', 'warning_refs'],
                )
            ],
            tmp_path,
        )
        assert task._restore_table_names() == ['drug_warning', 'studies', 'warning_refs']

    def test_deduplicates_a_table_named_twice(self, tmp_path: Path) -> None:
        task = self._task(
            [STUDIES],
            [QuerySpec(query='chembl_drug_warning', destination='d.parquet', requires_tables=['studies'])],
            tmp_path,
        )
        assert task._restore_table_names() == ['studies']

    def test_queries_only(self, tmp_path: Path) -> None:
        task = self._task(
            [],
            [QuerySpec(query='chembl_drug_warning', destination='d.parquet', requires_tables=['drug_warning'])],
            tmp_path,
        )
        assert task._restore_table_names() == ['drug_warning']
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -k RestoreTableNames -rxs`
Expected: FAIL with `AttributeError: 'PostgresExport' object has no attribute '_restore_table_names'`

- [ ] **Step 3: Add the method and use it**

Add to `PostgresExport`, immediately above `_restore`:

```python
    def _restore_table_names(self) -> list[str]:
        """Every table the restore must load, for table exports and queries alike."""
        names = {t.table for t in self.spec.tables}
        for query in self.spec.queries:
            names.update(query.requires_tables)
        return sorted(names)
```

In `_restore`, replace the first line:

```python
        tables = [t.table for t in self.spec.tables]
```

with:

```python
        tables = self._restore_table_names()
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -rxs`
Expected: PASS

- [ ] **Step 5: Lint, type-check and commit**

```bash
uv run --frozen --directory pis ruff check src tests
uv run --frozen --directory pis ty check src tests
git add pis/src/pis/tasks/postgres_export.py pis/tests/test_postgres_export.py
git commit -m "pis: restore the tables postgres_export queries need"
```

---

### Task 3: Query export execution

**Files:**
- Modify: `pis/src/pis/tasks/postgres_export.py:175-200` (helpers), `:355-400` (`_export`), `:425-455` (`validate`)
- Test: `pis/tests/test_postgres_export.py`

**Interfaces:**
- Consumes: `_load_query`, `QuerySpec`, `_sql_str`, `_scalar`, `self.row_counts` (existing).
- Produces: `_build_query_copy_sql(sql: str, destination: Path) -> str`; `PostgresExport._export_query(self, con: duckdb.DuckDBPyConnection, query: QuerySpec) -> Artifact`.

- [ ] **Step 1: Write the failing unit test for the SQL builder**

Append to `pis/tests/test_postgres_export.py`:

```python
class TestBuildQueryCopySql:
    def test_wraps_the_query(self) -> None:
        sql = _build_query_copy_sql('SELECT 1 AS a', Path('/work/out.parquet'))
        assert sql == "COPY (SELECT 1 AS a) TO '/work/out.parquet' (FORMAT parquet, COMPRESSION zstd)"

    def test_strips_a_trailing_semicolon(self) -> None:
        sql = _build_query_copy_sql('SELECT 1 AS a;\n', Path('/o.parquet'))
        assert sql.startswith('COPY (SELECT 1 AS a)')

    def test_does_not_deduplicate(self) -> None:
        assert 'DISTINCT' not in _build_query_copy_sql('SELECT 1 AS a', Path('/o.parquet'))
```

The third test matters: table exports apply `SELECT DISTINCT`, and query exports must not, because the queries control their own grain.

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -k BuildQueryCopySql -rxs`
Expected: FAIL with `ImportError: cannot import name '_build_query_copy_sql'`

- [ ] **Step 3: Add the builder**

Add after `_build_copy_sql` in `pis/src/pis/tasks/postgres_export.py`:

```python
def _build_query_copy_sql(sql: str, destination: Path) -> str:
    """Wrap a query from a SQL file in a COPY that writes it to parquet.

    Unlike :py:func:`_build_copy_sql` this does not add ``DISTINCT``: a query
    file is responsible for its own grain.
    """
    body = sql.strip().rstrip(';')
    return f'COPY ({body}) TO {_sql_str(str(destination))} (FORMAT parquet, COMPRESSION zstd)'
```

Add `_build_query_copy_sql` to the test module's import block.

- [ ] **Step 4: Run it to verify it passes**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -k BuildQueryCopySql -rxs`
Expected: PASS

- [ ] **Step 5: Write the failing round-trip test**

Append to `pis/tests/test_postgres_export.py`, inside the `@pytest.mark.pgserver` block. Add a module-level constant near `DESTINATION`:

```python
QUERY_DESTINATION = 'out/q.parquet'
```

Then add this class after `TestRoundTrip`:

```python
@pytest.mark.pgserver
class TestQueryRoundTrip:
    """Run a SQL file against a real restored database and export the result."""

    def _build(self, work_path: Path, source: str) -> PostgresExport:
        spec = PostgresExportSpec(
            name='postgres_export demo query',
            source=source,
            schema_name='demo',
            queries=[
                QuerySpec(
                    query='_test_demo',
                    destination=QUERY_DESTINATION,
                    requires_tables=['t'],
                )
            ],
        )
        config = Config(step='demo', steps=['demo'], work_path=work_path, pool_size=2, log_level='DEBUG')
        context = TaskContext(config=config, scratchpad=Scratchpad())
        context.abort = Event()
        return PostgresExport(spec, context)

    def test_exports_a_query(self, dump: Path, tmp_path: Path) -> None:
        work = tmp_path / 'work'
        task = self._build(work, str(dump))
        _await(task.run())
        assert task.manifest.result != Result.FAILURE, task.manifest.failure_reason
        _await(task.validate())
        assert task.manifest.result == Result.SUCCESS, task.manifest.failure_reason

        out = work / QUERY_DESTINATION
        # the demo table has 1000 rows over 7 distinct txt values
        assert _count(out) == (7,)

    def test_a_failing_query_names_itself(self, dump: Path, tmp_path: Path) -> None:
        task = self._build(tmp_path / 'work', str(dump))
        task.spec.queries[0].requires_tables = ['no_such_table']
        _await(task.run())
        assert task.manifest.result == Result.FAILURE
        assert task.manifest.failure_reason is not None
        assert '_test_demo' in task.manifest.failure_reason
```

Move the `dump` fixture from `TestRoundTrip` to module scope so both classes can use it — change `@pytest.fixture(scope='class')` to `@pytest.fixture(scope='module')` and lift it out of the class body, dropping the `self` parameter.

`pis/tests/sql/_test_demo.sql` already exists from Task 1. `TestQueryRoundTrip._build` must point the task at it by setting `QUERY_PACKAGE`, since the task loads queries from `pis.sql` by default. Add to `_build`, before constructing the spec:

```python
        monkeypatch.setattr(postgres_export, 'QUERY_PACKAGE', 'tests.sql')
```

and thread `monkeypatch: pytest.MonkeyPatch` through `_build` and both test methods. Spec validation calls `_load_query`, so the patch must be in place before `PostgresExportSpec(...)` is constructed.

- [ ] **Step 6: Run it to verify it fails**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -k QueryRoundTrip -rxs`
Expected: FAIL — the parquet is never written, because `_export` ignores `queries`.

- [ ] **Step 7: Implement query export**

Add to `PostgresExport`, after `_export_table`:

```python
    def _export_query(self, con: duckdb.DuckDBPyConnection, query: QuerySpec) -> Artifact:
        sql = _load_query(query.query)

        local = Path(StorageHandle(query.destination, config=self.context.config, force_local=True).absolute)
        local.parent.mkdir(parents=True, exist_ok=True)

        try:
            rows = int(_scalar(con, _build_query_copy_sql(sql, local)))
        except duckdb.Error as e:
            # duckdb names the missing relation but not the query, and a config
            # with several queries gives no other clue which one failed
            raise PostgresExportError(f'query {query.query} failed: {e}')

        if not rows:
            raise PostgresExportError(f'query {query.query} exported no rows')

        logger.info(f'exported {rows} rows from query {query.query} to {query.destination}')
        self.row_counts[query.destination] = rows

        dst = StorageHandle(query.destination, config=self.context.config)
        if dst.absolute != str(local):
            logger.debug(f'uploading {local} to {dst.absolute}')
            with local.open('rb') as f, dst.open('wb') as g:
                self._copy_stream(f, g)

        return Artifact(source=f'{self.spec.source}#{query.query}', destination=dst.absolute)
```

In `_export`, after the existing table loop and before `return artifacts`:

```python
            if self.spec.queries:
                # the SQL files use bare table names; point the default catalog at
                # the attached server so they do not have to spell out pg."schema"
                con.execute(f'USE pg.{_quote_ident(self.spec.schema_name)}')
                for query in self.spec.queries:
                    self._check_abort()
                    artifacts.append(self._export_query(con, query))
```

Add the identifier quoter beside `_sql_str`:

```python
def _quote_ident(name: str) -> str:
    """Render a python string as a quoted SQL identifier."""
    escaped = name.replace('"', '""')
    return f'"{escaped}"'
```

In `validate`, change the loop header so it covers both lists:

```python
            destinations = [t.destination for t in self.spec.tables] + [q.destination for q in self.spec.queries]
            for destination in destinations:
```

and replace every `table.destination` inside that loop body with `destination`.

- [ ] **Step 8: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_postgres_export.py -rxs`
Expected: PASS, all tests.

- [ ] **Step 9: Lint, type-check and commit**

```bash
uv run --frozen --directory pis ruff check src tests
uv run --frozen --directory pis ty check src tests
git add pis/src/pis pis/tests/test_postgres_export.py
git commit -m "pis: export postgres_export queries to parquet"
```

---

### Task 4: Comparison harness

**Files:**
- Create: `/tmp/chembl-baseline/compare_chembl_es.py` — **outside the repository. Never `git add` this file.**

**Interfaces:**
- Consumes: nothing from earlier tasks; it reads files.
- Produces: a CLI `uv run --directory pis python /tmp/chembl-baseline/compare_chembl_es.py <dataset> <parquet> <baseline.jsonl>` that exits non-zero on any difference. Tasks 5-8 each run it.

This task has no unit tests — it is the test harness. Its correctness is established by running it against a baseline compared with itself in Step 3.

This task has **no commit**. The harness is a development tool whose output is the deliverable, not the file itself. If `git status` ever lists it, it is in the wrong place — move it out of the working tree.

- [ ] **Step 1: Write the script**

Run `mkdir -p /tmp/chembl-baseline` first, then create `/tmp/chembl-baseline/compare_chembl_es.py`:

```python
"""Compare a rebuilt ChEMBL parquet against the Elasticsearch JSONL it replaces.

Usage:
    python /tmp/chembl-baseline/compare_chembl_es.py <dataset> <parquet> <baseline.jsonl>

``dataset`` is one of chembl_molecule, chembl_mechanism, chembl_target,
chembl_drug_warning. Exits 1 if the two differ.
"""

import json
import sys
from collections import Counter
from typing import Any

import duckdb

KEYS = {
    'chembl_molecule': ['molecule_chembl_id'],
    'chembl_mechanism': ['record_id', 'molecule_chembl_id', 'target_chembl_id', 'mechanism_of_action'],
    'chembl_target': ['target_chembl_id'],
    'chembl_drug_warning': ['warning_id'],
}

# fields deliberately dropped as unread by pts; see the design doc
PRUNED = {
    'chembl_molecule': {'first_approval', 'max_phase', 'withdrawn_flag', 'black_box_warning'},
    'chembl_mechanism': set(),
    'chembl_target': set(),
    'chembl_drug_warning': set(),
}


def _normalise(value: Any) -> Any:
    """Make two representations of the same value compare equal.

    Lists of structs come out of parquet in a fixed order and out of ES in an
    arbitrary one, and parquet gives an empty list where ES sometimes gives null.
    """
    if isinstance(value, dict):
        return tuple(sorted((k, _normalise(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple)):
        return tuple(sorted((repr(_normalise(v)) for v in value)))
    return value


def _load_baseline(path: str, dataset: str) -> dict[tuple, dict]:
    decoder = json.JSONDecoder(strict=False)
    rows = {}
    for line in open(path, errors='replace'):
        line = line.strip()
        if not line.startswith('{'):
            continue
        doc = decoder.decode(line)
        key = tuple(_flat(doc, k) for k in KEYS[dataset])
        rows[key] = doc
    return rows


def _flat(doc: dict, path: str) -> Any:
    cur: Any = doc
    for part in path.split('.'):
        cur = (cur or {}).get(part)
    return cur


def _load_parquet(path: str, dataset: str) -> dict[tuple, dict]:
    con = duckdb.connect()
    con.execute(f"CREATE VIEW v AS SELECT * FROM read_parquet('{path}')")
    columns = [r[0] for r in con.execute('DESCRIBE v').fetchall()]
    rows = {}
    for record in con.execute('SELECT * FROM v').fetchall():
        doc = dict(zip(columns, record, strict=True))
        key = tuple(_flat(doc, k) for k in KEYS[dataset])
        rows[key] = doc
    return rows


def main() -> int:
    dataset, parquet, baseline = sys.argv[1], sys.argv[2], sys.argv[3]
    if dataset not in KEYS:
        print(f'unknown dataset {dataset}; expected one of {sorted(KEYS)}')
        return 2

    new = _load_parquet(parquet, dataset)
    old = _load_baseline(baseline, dataset)

    print(f'{dataset}: {len(new)} parquet rows, {len(old)} baseline rows')
    failed = False

    only_new, only_old = set(new) - set(old), set(old) - set(new)
    if only_new or only_old:
        failed = True
        print(f'  KEY MISMATCH: {len(only_new)} only in parquet, {len(only_old)} only in baseline')
        for k in list(only_new)[:5]:
            print(f'    only in parquet:  {k}')
        for k in list(only_old)[:5]:
            print(f'    only in baseline: {k}')

    expected_fields = (set().union(*(d.keys() for d in old.values())) if old else set()) - PRUNED[dataset]
    actual_fields = set().union(*(d.keys() for d in new.values())) if new else set()
    if expected_fields - actual_fields:
        failed = True
        print(f'  MISSING FIELDS: {sorted(expected_fields - actual_fields)}')
    if actual_fields - expected_fields:
        failed = True
        print(f'  UNEXPECTED FIELDS: {sorted(actual_fields - expected_fields)}')

    mismatches: Counter = Counter()
    samples: dict[str, tuple] = {}
    for key in set(new) & set(old):
        for field in expected_fields & actual_fields:
            a, b = _normalise(new[key].get(field)), _normalise(old[key].get(field))
            if a != b:
                mismatches[field] += 1
                samples.setdefault(field, (key, new[key].get(field), old[key].get(field)))

    for field, count in mismatches.most_common():
        failed = True
        key, got, want = samples[field]
        print(f'  FIELD {field}: {count} mismatches, e.g. key={key}')
        print(f'    parquet:  {str(got)[:200]}')
        print(f'    baseline: {str(want)[:200]}')

    print('  MATCH' if not failed else '  DIFFERENCES FOUND')
    return 1 if failed else 0


if __name__ == '__main__':
    sys.exit(main())
```

- [ ] **Step 2: Download the baselines**

```bash
mkdir -p /tmp/chembl-baseline
for f in chembl_molecule chembl_mechanism chembl_target chembl_drug_warning; do
  gcloud storage cp "gs://open-targets-data-releases/26.06/input/drug/$f.jsonl" "/tmp/chembl-baseline/$f.jsonl"
done
```

These are large. Confirm `gcloud auth login` is current first.

- [ ] **Step 3: Sanity-check the harness against itself**

Convert one baseline to parquet and compare it with its own source. A harness that reports differences here is broken.

```bash
uv run --frozen --directory pis python -c "
import duckdb
duckdb.connect().execute(\"COPY (SELECT * FROM read_json_auto('/tmp/chembl-baseline/chembl_drug_warning.jsonl')) TO '/tmp/chembl-baseline/self.parquet' (FORMAT parquet)\")
"
uv run --frozen --directory pis python /tmp/chembl-baseline/compare_chembl_es.py \
  chembl_drug_warning /tmp/chembl-baseline/self.parquet /tmp/chembl-baseline/chembl_drug_warning.jsonl
```

Expected: `MATCH`, exit 0. If it reports differences, fix `_normalise` before going further — every later task depends on this.

- [ ] **Step 4: Confirm nothing was added to the repository**

Run: `git status --porcelain`
Expected: no line mentioning `compare_chembl_es.py`. There is deliberately no commit in this task.

---

### Task 5: `chembl_drug_warning.sql`

**Files:**
- Create: `pis/src/pis/sql/chembl_drug_warning.sql`
- Create: `pis/tests/test_chembl_queries.py`

**Interfaces:**
- Consumes: the query machinery from Tasks 1-3.
- Produces: a fixture-database helper `chembl_fixture(tmp_path_factory)` in `test_chembl_queries.py`, reused by Tasks 6-8, which creates a small ChEMBL-shaped schema and returns a `PostgresServer`.

- [ ] **Step 1: Write the failing test**

Create `pis/tests/test_chembl_queries.py`:

```python
"""Run the ChEMBL rebuild queries against a small fixture database."""

import duckdb
import pytest
from pixeltable_pgserver.postgres_server import PostgresServer, get_server

from pis.tasks.postgres_export import _load_query

pytestmark = pytest.mark.pgserver

SCHEMA = """
CREATE TABLE molecule_dictionary (molregno int PRIMARY KEY, chembl_id text, pref_name text, molecule_type text);
CREATE TABLE molecule_hierarchy (molregno int PRIMARY KEY, parent_molregno int, active_molregno int);
CREATE TABLE drug_warning (
    warning_id int PRIMARY KEY, record_id int, molregno int, warning_type text, warning_class text,
    warning_country text, warning_description text, warning_year int,
    efo_term text, efo_id text, efo_id_for_warning_class text
);
CREATE TABLE warning_refs (warnref_id int PRIMARY KEY, warning_id int, ref_type text, ref_id text, ref_url text);
"""

DATA = """
INSERT INTO molecule_dictionary VALUES
    (1, 'CHEMBL1', 'child drug', 'Small molecule'),
    (2, 'CHEMBL2', 'parent drug', 'Small molecule'),
    (3, 'CHEMBL3', 'lone drug', 'Small molecule');
INSERT INTO molecule_hierarchy VALUES (1, 2, 2), (2, 2, 2), (3, 3, 3);
INSERT INTO drug_warning VALUES
    (10, 100, 1, 'Withdrawn', 'Cardiotoxicity', 'France', 'bad things', 2009, 'term', 'EFO_1', 'EFO_2'),
    (11, 101, 3, 'Warning', NULL, 'US', NULL, NULL, NULL, NULL, NULL);
INSERT INTO warning_refs VALUES
    (1, 10, 'ISBN', 'ref-a', 'http://a'),
    (2, 10, 'DOI', 'ref-b', 'http://b');
"""


@pytest.fixture(scope='module')
def chembl(tmp_path_factory: pytest.TempPathFactory) -> PostgresServer:
    """A postgres server holding a miniature ChEMBL."""
    server = get_server(tmp_path_factory.mktemp('chembl') / 'pgdata', cleanup_mode='delete')
    server.psql(SCHEMA)
    server.psql(DATA)
    return server


def run_query(server: PostgresServer, name: str) -> list[dict]:
    """Run a shipped SQL file against the fixture database and return rows as dicts."""
    con = duckdb.connect()
    con.execute('LOAD postgres')
    con.execute(f"ATTACH '{server.get_uri(database='postgres')}' AS pg (TYPE postgres, READ_ONLY)")
    con.execute('USE pg."public"')
    result = con.execute(_load_query(name))
    columns = [d[0] for d in result.description]
    return [dict(zip(columns, row, strict=True)) for row in result.fetchall()]


class TestDrugWarning:
    @pytest.fixture(scope='class')
    def rows(self, chembl: PostgresServer) -> dict[int, dict]:
        return {r['warning_id']: r for r in run_query(chembl, 'chembl_drug_warning')}

    def test_one_row_per_warning(self, rows: dict[int, dict]) -> None:
        assert sorted(rows) == [10, 11]

    def test_scalar_fields(self, rows: dict[int, dict]) -> None:
        w = rows[10]
        assert w['warning_type'] == 'Withdrawn'
        assert w['warning_class'] == 'Cardiotoxicity'
        assert w['warning_country'] == 'France'
        assert w['warning_description'] == 'bad things'
        assert w['warning_year'] == 2009
        assert w['efo_id'] == 'EFO_1'
        assert w['efo_term'] == 'term'
        assert w['efo_id_for_warning_class'] == 'EFO_2'

    def test_molecule_and_parent(self, rows: dict[int, dict]) -> None:
        assert rows[10]['molecule_chembl_id'] == 'CHEMBL1'
        assert rows[10]['parent_molecule_chembl_id'] == 'CHEMBL2'

    def test_all_molecule_chembl_ids_has_both(self, rows: dict[int, dict]) -> None:
        assert sorted(rows[10]['_metadata']['all_molecule_chembl_ids']) == ['CHEMBL1', 'CHEMBL2']

    def test_all_molecule_chembl_ids_deduplicates(self, rows: dict[int, dict]) -> None:
        # CHEMBL3 is its own parent
        assert rows[11]['_metadata']['all_molecule_chembl_ids'] == ['CHEMBL3']

    def test_refs(self, rows: dict[int, dict]) -> None:
        refs = rows[10]['warning_refs']
        assert len(refs) == 2
        assert {r['ref_type'] for r in refs} == {'ISBN', 'DOI'}
        assert {r['ref_id'] for r in refs} == {'ref-a', 'ref-b'}

    def test_no_refs_is_an_empty_list_not_null(self, rows: dict[int, dict]) -> None:
        assert rows[11]['warning_refs'] == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -rxs`
Expected: FAIL with `PostgresExportError: no SQL file for query chembl_drug_warning` — the file does not exist yet.

- [ ] **Step 3: Write the query**

Replace `pis/src/pis/sql/chembl_drug_warning.sql` entirely:

```sql
-- Rebuild the chembl_<version>_drug_warning Elasticsearch document from the
-- ChEMBL relational schema.
--
-- Grain: one row per drug_warning.warning_id.
-- See docs/superpowers/specs/2026-08-10-chembl-postgres-drug-extraction-design.md
WITH parent AS (
    SELECT mh.molregno,
           pmd.chembl_id AS parent_chembl_id
    FROM molecule_hierarchy mh
    JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
),
refs AS (
    SELECT wr.warning_id,
           list(struct_pack(
               ref_id := wr.ref_id,
               ref_type := wr.ref_type,
               ref_url := wr.ref_url
           ) ORDER BY wr.warnref_id) AS warning_refs
    FROM warning_refs wr
    GROUP BY wr.warning_id
)
SELECT
    dw.warning_id,
    dw.warning_type,
    dw.warning_class,
    dw.warning_country,
    dw.warning_description,
    dw.warning_year,
    dw.efo_id,
    dw.efo_term,
    dw.efo_id_for_warning_class,
    md.chembl_id AS molecule_chembl_id,
    p.parent_chembl_id AS parent_molecule_chembl_id,
    struct_pack(
        all_molecule_chembl_ids := list_distinct(
            list_filter([md.chembl_id, p.parent_chembl_id], x -> x IS NOT NULL)
        )
    ) AS _metadata,
    coalesce(
        r.warning_refs,
        []::STRUCT(ref_id VARCHAR, ref_type VARCHAR, ref_url VARCHAR)[]
    ) AS warning_refs
FROM drug_warning dw
LEFT JOIN molecule_dictionary md ON md.molregno = dw.molregno
LEFT JOIN parent p ON p.molregno = dw.molregno
LEFT JOIN refs r ON r.warning_id = dw.warning_id
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -rxs`
Expected: PASS

- [ ] **Step 5: Compare against the real baseline**

This needs the real ChEMBL dump restored. Run the task end to end against it with a local `work_path`, then:

```bash
uv run --frozen --directory pis python /tmp/chembl-baseline/compare_chembl_es.py \
  chembl_drug_warning <work_path>/input/drug/chembl_drug_warning.parquet \
  /tmp/chembl-baseline/chembl_drug_warning.jsonl
```

Expected: `MATCH`. If not, adjust the query until it matches and note what changed. Record the final output in the PR description.

- [ ] **Step 6: Lint, type-check and commit**

```bash
uv run --frozen --directory pis ruff check src tests
uv run --frozen --directory pis ty check src tests
git add pis/src/pis/sql/chembl_drug_warning.sql pis/tests/test_chembl_queries.py
git commit -m "pis: rebuild chembl drug warnings from postgres"
```

---

### Task 6: `chembl_mechanism.sql`

**Files:**
- Create: `pis/src/pis/sql/chembl_mechanism.sql`
- Modify: `pis/tests/test_chembl_queries.py`

**Interfaces:**
- Consumes: the `chembl` fixture and `run_query` from Task 5.
- Produces: nothing new.

- [ ] **Step 1: Extend the fixture and write the failing test**

In `pis/tests/test_chembl_queries.py`, append to `SCHEMA`:

```python
SCHEMA += """
CREATE TABLE target_dictionary (tid int PRIMARY KEY, target_type text, pref_name text, chembl_id text);
CREATE TABLE drug_mechanism (
    mec_id int PRIMARY KEY, record_id int, molregno int, mechanism_of_action text, tid int, action_type text
);
CREATE TABLE mechanism_refs (mecref_id int PRIMARY KEY, mec_id int, ref_type text, ref_id text, ref_url text);
"""
```

and append to `DATA`:

```python
DATA += """
INSERT INTO target_dictionary VALUES (500, 'SINGLE PROTEIN', 'A target', 'CHEMBL_T1');
INSERT INTO drug_mechanism VALUES
    (20, 200, 1, 'Kinase inhibitor', 500, 'INHIBITOR'),
    (21, 201, 3, 'Receptor agonist', NULL, NULL);
INSERT INTO mechanism_refs VALUES (1, 20, 'PubMed', '12345', 'http://pm/12345');
"""
```

Add the test class:

```python
class TestMechanism:
    @pytest.fixture(scope='class')
    def rows(self, chembl: PostgresServer) -> dict[int, dict]:
        return {r['record_id']: r for r in run_query(chembl, 'chembl_mechanism')}

    def test_one_row_per_mechanism(self, rows: dict[int, dict]) -> None:
        assert sorted(rows) == [200, 201]

    def test_scalar_fields(self, rows: dict[int, dict]) -> None:
        m = rows[200]
        assert m['mechanism_of_action'] == 'Kinase inhibitor'
        assert m['action_type'] == 'INHIBITOR'
        assert m['molecule_chembl_id'] == 'CHEMBL1'
        assert m['parent_molecule_chembl_id'] == 'CHEMBL2'

    def test_target(self, rows: dict[int, dict]) -> None:
        assert rows[200]['target_chembl_id'] == 'CHEMBL_T1'

    def test_missing_target_is_null(self, rows: dict[int, dict]) -> None:
        assert rows[201]['target_chembl_id'] is None

    def test_all_molecule_chembl_ids(self, rows: dict[int, dict]) -> None:
        assert sorted(rows[200]['_metadata']['all_molecule_chembl_ids']) == ['CHEMBL1', 'CHEMBL2']

    def test_refs(self, rows: dict[int, dict]) -> None:
        refs = rows[200]['mechanism_refs']
        assert len(refs) == 1
        assert refs[0]['ref_type'] == 'PubMed'
        assert refs[0]['ref_id'] == '12345'

    def test_no_refs_is_an_empty_list_not_null(self, rows: dict[int, dict]) -> None:
        assert rows[201]['mechanism_refs'] == []
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -k Mechanism -rxs`
Expected: FAIL with `PostgresExportError: no SQL file for query chembl_mechanism`

- [ ] **Step 3: Write the query**

Create `pis/src/pis/sql/chembl_mechanism.sql`:

```sql
-- Rebuild the chembl_<version>_mechanism Elasticsearch document from the ChEMBL
-- relational schema.
--
-- Grain: one row per drug_mechanism.mec_id.
WITH parent AS (
    SELECT mh.molregno,
           pmd.chembl_id AS parent_chembl_id
    FROM molecule_hierarchy mh
    JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
),
refs AS (
    SELECT mr.mec_id,
           list(struct_pack(
               ref_id := mr.ref_id,
               ref_type := mr.ref_type,
               ref_url := mr.ref_url
           ) ORDER BY mr.mecref_id) AS mechanism_refs
    FROM mechanism_refs mr
    GROUP BY mr.mec_id
)
SELECT
    dm.record_id,
    dm.mechanism_of_action,
    dm.action_type,
    md.chembl_id AS molecule_chembl_id,
    p.parent_chembl_id AS parent_molecule_chembl_id,
    td.chembl_id AS target_chembl_id,
    struct_pack(
        all_molecule_chembl_ids := list_distinct(
            list_filter([md.chembl_id, p.parent_chembl_id], x -> x IS NOT NULL)
        )
    ) AS _metadata,
    coalesce(
        r.mechanism_refs,
        []::STRUCT(ref_id VARCHAR, ref_type VARCHAR, ref_url VARCHAR)[]
    ) AS mechanism_refs
FROM drug_mechanism dm
LEFT JOIN molecule_dictionary md ON md.molregno = dm.molregno
LEFT JOIN parent p ON p.molregno = dm.molregno
LEFT JOIN target_dictionary td ON td.tid = dm.tid
LEFT JOIN refs r ON r.mec_id = dm.mec_id
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -rxs`
Expected: PASS

- [ ] **Step 5: Compare against the real baseline**

```bash
uv run --frozen --directory pis python /tmp/chembl-baseline/compare_chembl_es.py \
  chembl_mechanism <work_path>/input/drug/chembl_mechanism.parquet \
  /tmp/chembl-baseline/chembl_mechanism.jsonl
```

Expected: `MATCH`. The row count also settles whether the ES grain really is `mec_id`; if the counts differ, that assumption is wrong and the query needs revisiting before you continue.

- [ ] **Step 6: Lint, type-check and commit**

```bash
uv run --frozen --directory pis ruff check src tests
uv run --frozen --directory pis ty check src tests
git add pis/src/pis/sql/chembl_mechanism.sql pis/tests/test_chembl_queries.py
git commit -m "pis: rebuild chembl mechanisms from postgres"
```

---

### Task 7: `chembl_molecule.sql`

**Files:**
- Create: `pis/src/pis/sql/chembl_molecule.sql`
- Modify: `pis/tests/test_chembl_queries.py`

**Interfaces:**
- Consumes: the `chembl` fixture and `run_query` from Task 5.
- Produces: nothing new.

This task resolves open question 1 from the spec: how `cross_references` is built. The starting hypothesis below is `compound_records.src_compound_id` joined to `source.src_short_name`, excluding `src_id = 1` (scientific literature). Step 5 confirms or corrects it.

- [ ] **Step 1: Extend the fixture and write the failing test**

Append to `SCHEMA`:

```python
SCHEMA += """
CREATE TABLE compound_structures (
    molregno int PRIMARY KEY, molfile text, standard_inchi text, standard_inchi_key text, canonical_smiles text
);
CREATE TABLE molecule_synonyms (molsyn_id int PRIMARY KEY, molregno int, syn_type text, synonyms text);
CREATE TABLE source (src_id int PRIMARY KEY, src_short_name text, src_description text);
CREATE TABLE compound_records (
    record_id int PRIMARY KEY, molregno int, doc_id int, src_id int, src_compound_id text
);
"""
```

Append to `DATA`:

```python
DATA += """
INSERT INTO compound_structures VALUES (1, 'MOLBLOCK1', 'InChI=1S/x', 'INCHIKEY1', 'CCO');
INSERT INTO compound_structures VALUES (2, 'MOLBLOCK2', 'InChI=1S/y', 'INCHIKEY2', 'CCC');
INSERT INTO molecule_synonyms VALUES
    (1, 1, 'TRADE_NAME', 'Tradey'),
    (2, 1, 'INN', 'childium');
INSERT INTO source VALUES (1, 'LITERATURE', 'Scientific Literature'), (7, 'DRUGBANK', 'DrugBank');
INSERT INTO compound_records VALUES
    (100, 1, 900, 7, 'DB00001'),
    (101, 1, 901, 1, 'IGNORED'),
    (102, 3, 902, 7, NULL);
"""
```

Add the test class:

```python
class TestMolecule:
    @pytest.fixture(scope='class')
    def rows(self, chembl: PostgresServer) -> dict[str, dict]:
        return {r['molecule_chembl_id']: r for r in run_query(chembl, 'chembl_molecule')}

    def test_one_row_per_molecule(self, rows: dict[str, dict]) -> None:
        assert sorted(rows) == ['CHEMBL1', 'CHEMBL2', 'CHEMBL3']

    def test_scalar_fields(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL1']['pref_name'] == 'child drug'
        assert rows['CHEMBL1']['molecule_type'] == 'Small molecule'

    def test_structures(self, rows: dict[str, dict]) -> None:
        s = rows['CHEMBL1']['molecule_structures']
        assert s['canonical_smiles'] == 'CCO'
        assert s['standard_inchi_key'] == 'INCHIKEY1'
        assert s['molfile'] == 'MOLBLOCK1'

    def test_standard_inchi_is_pruned(self, rows: dict[str, dict]) -> None:
        assert 'standard_inchi' not in rows['CHEMBL1']['molecule_structures']

    def test_missing_structures_is_null(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL3']['molecule_structures'] is None

    def test_hierarchy_has_only_parent(self, rows: dict[str, dict]) -> None:
        h = rows['CHEMBL1']['molecule_hierarchy']
        assert h['parent_chembl_id'] == 'CHEMBL2'
        assert set(h) == {'parent_chembl_id'}

    def test_synonyms(self, rows: dict[str, dict]) -> None:
        syns = rows['CHEMBL1']['molecule_synonyms']
        assert {s['molecule_synonym'] for s in syns} == {'Tradey', 'childium'}
        assert {s['syn_type'] for s in syns} == {'TRADE_NAME', 'INN'}

    def test_no_synonyms_is_an_empty_list_not_null(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL3']['molecule_synonyms'] == []

    def test_cross_references_excludes_literature(self, rows: dict[str, dict]) -> None:
        xrefs = rows['CHEMBL1']['cross_references']
        assert xrefs == [{'xref_id': 'DB00001', 'xref_src': 'DRUGBANK'}]

    def test_cross_references_skips_null_ids(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL3']['cross_references'] == []

    def test_dead_fields_are_absent(self, rows: dict[str, dict]) -> None:
        for dead in ('first_approval', 'max_phase', 'withdrawn_flag', 'black_box_warning'):
            assert dead not in rows['CHEMBL1']
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -k Molecule -rxs`
Expected: FAIL with `PostgresExportError: no SQL file for query chembl_molecule`

- [ ] **Step 3: Write the query**

Create `pis/src/pis/sql/chembl_molecule.sql`:

```sql
-- Rebuild the chembl_<version>_molecule Elasticsearch document from the ChEMBL
-- relational schema.
--
-- Grain: one row per molecule_dictionary.molregno.
--
-- Note the two name differences against the Elasticsearch document: the
-- `molecule_structures` field comes from the `compound_structures` table, and
-- `molecule_synonym` comes from the `synonyms` column.
--
-- first_approval, max_phase, withdrawn_flag and black_box_warning are
-- deliberately absent: no pts module reads them.
WITH synonyms AS (
    SELECT ms.molregno,
           list(struct_pack(
               molecule_synonym := ms.synonyms,
               syn_type := ms.syn_type
           ) ORDER BY ms.molsyn_id) AS molecule_synonyms
    FROM molecule_synonyms ms
    GROUP BY ms.molregno
),
xrefs AS (
    -- ChEMBL has no compound xref table; the Elasticsearch document's
    -- cross_references are the non-literature source records for the molecule
    SELECT cr.molregno,
           list(DISTINCT struct_pack(
               xref_id := cr.src_compound_id,
               xref_src := s.src_short_name
           )) AS cross_references
    FROM compound_records cr
    JOIN source s ON s.src_id = cr.src_id
    WHERE cr.src_compound_id IS NOT NULL
      AND cr.src_id <> 1
    GROUP BY cr.molregno
),
parent AS (
    SELECT mh.molregno,
           pmd.chembl_id AS parent_chembl_id
    FROM molecule_hierarchy mh
    JOIN molecule_dictionary pmd ON pmd.molregno = mh.parent_molregno
)
SELECT
    md.chembl_id AS molecule_chembl_id,
    md.pref_name,
    md.molecule_type,
    CASE WHEN cs.molregno IS NULL THEN NULL ELSE struct_pack(
        canonical_smiles := cs.canonical_smiles,
        standard_inchi_key := cs.standard_inchi_key,
        molfile := cs.molfile
    ) END AS molecule_structures,
    struct_pack(parent_chembl_id := p.parent_chembl_id) AS molecule_hierarchy,
    coalesce(
        sy.molecule_synonyms,
        []::STRUCT(molecule_synonym VARCHAR, syn_type VARCHAR)[]
    ) AS molecule_synonyms,
    coalesce(
        x.cross_references,
        []::STRUCT(xref_id VARCHAR, xref_src VARCHAR)[]
    ) AS cross_references
FROM molecule_dictionary md
LEFT JOIN compound_structures cs ON cs.molregno = md.molregno
LEFT JOIN parent p ON p.molregno = md.molregno
LEFT JOIN synonyms sy ON sy.molregno = md.molregno
LEFT JOIN xrefs x ON x.molregno = md.molregno
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -rxs`
Expected: PASS

- [ ] **Step 5: Compare against the real baseline and settle `cross_references`**

```bash
uv run --frozen --directory pis python /tmp/chembl-baseline/compare_chembl_es.py \
  chembl_molecule <work_path>/input/drug/chembl_molecule.parquet \
  /tmp/chembl-baseline/chembl_molecule.jsonl
```

If `cross_references` mismatches, the harness prints a sample of the parquet and baseline values for the same molecule. Use it to work out the real rule — likely candidates are a different `src_id` exclusion set, or `compound_records.compound_key` instead of `src_compound_id`. Iterate until `MATCH`, then update both the query comment and the spec's open question 1 with the answer.

- [ ] **Step 6: Lint, type-check and commit**

```bash
uv run --frozen --directory pis ruff check src tests
uv run --frozen --directory pis ty check src tests
git add pis/src/pis/sql/chembl_molecule.sql pis/tests/test_chembl_queries.py docs/superpowers/specs
git commit -m "pis: rebuild chembl molecules from postgres"
```

---

### Task 8: `chembl_target.sql`

**Files:**
- Create: `pis/src/pis/sql/chembl_target.sql`
- Modify: `pis/tests/test_chembl_queries.py`

**Interfaces:**
- Consumes: the `chembl` fixture and `run_query` from Task 5.
- Produces: nothing new.

This is the hardest query. Two things must hold: `_metadata.protein_classification` stays positionally aligned with `target_components`, because `pts/src/pts/pyspark/target.py:1150` zips them; and each component contributes exactly one classification entry. This task also resolves open question 2 — which class wins when a component has several. The starting rule below is deepest `class_level`, ties broken by smallest `protein_class_id`.

- [ ] **Step 1: Extend the fixture and write the failing test**

Append to `SCHEMA`:

```python
SCHEMA += """
CREATE TABLE component_sequences (component_id int PRIMARY KEY, accession text, component_type text);
CREATE TABLE target_components (targcomp_id int PRIMARY KEY, tid int, component_id int, homologue int);
CREATE TABLE protein_classification (
    protein_class_id int PRIMARY KEY, parent_id int, pref_name text, short_name text, class_level int
);
CREATE TABLE component_class (comp_class_id int PRIMARY KEY, component_id int, protein_class_id int);
"""
```

Append to `DATA`:

```python
DATA += """
INSERT INTO target_dictionary VALUES
    (501, 'SINGLE PROTEIN', 'Single target', 'CHEMBL_T2'),
    (502, 'PROTEIN COMPLEX', 'Complex target', 'CHEMBL_T3'),
    (503, 'CELL-LINE', 'No components', 'CHEMBL_T4');
INSERT INTO component_sequences VALUES
    (1, 'P00001', 'PROTEIN'),
    (2, 'P00002', 'PROTEIN');
INSERT INTO target_components VALUES
    (1, 501, 1, 0),
    (2, 502, 1, 0),
    (3, 502, 2, 0);
INSERT INTO protein_classification VALUES
    (10, NULL, 'Enzyme', 'enz', 1),
    (11, 10, 'Kinase', 'kin', 2),
    (12, 11, 'Protein Kinase', 'pk', 3),
    (20, NULL, 'Transporter', 'tra', 1);
INSERT INTO component_class VALUES
    (1, 1, 12),
    (2, 1, 20),
    (3, 2, 11);
"""
```

Add the test class:

```python
class TestTarget:
    @pytest.fixture(scope='class')
    def rows(self, chembl: PostgresServer) -> dict[str, dict]:
        return {r['target_chembl_id']: r for r in run_query(chembl, 'chembl_target')}

    def test_one_row_per_target(self, rows: dict[str, dict]) -> None:
        assert sorted(rows) == ['CHEMBL_T1', 'CHEMBL_T2', 'CHEMBL_T3', 'CHEMBL_T4']

    def test_scalar_fields(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T2']['pref_name'] == 'Single target'
        assert rows['CHEMBL_T2']['target_type'] == 'SINGLE PROTEIN'

    def test_components_carry_only_accession(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T2']['target_components'] == [{'accession': 'P00001'}]

    def test_target_with_no_components_is_an_empty_list(self, rows: dict[str, dict]) -> None:
        assert rows['CHEMBL_T4']['target_components'] == []
        assert rows['CHEMBL_T4']['_metadata']['protein_classification'] == []

    def test_classification_is_aligned_with_components(self, rows: dict[str, dict]) -> None:
        for target in rows.values():
            assert len(target['_metadata']['protein_classification']) == len(target['target_components'])

    def test_multi_component_target_keeps_component_order(self, rows: dict[str, dict]) -> None:
        complex_target = rows['CHEMBL_T3']
        assert [c['accession'] for c in complex_target['target_components']] == ['P00001', 'P00002']

    def test_deepest_class_wins(self, rows: dict[str, dict]) -> None:
        # component 1 has both Protein Kinase (level 3) and Transporter (level 1)
        pc = rows['CHEMBL_T2']['_metadata']['protein_classification'][0]
        assert pc['protein_class_id'] == 12

    def test_ancestors_are_flattened_into_levels(self, rows: dict[str, dict]) -> None:
        pc = rows['CHEMBL_T2']['_metadata']['protein_classification'][0]
        assert pc['l1'] == 'Enzyme'
        assert pc['l2'] == 'Kinase'
        assert pc['l3'] == 'Protein Kinase'
        assert pc['l4'] is None
        assert pc['l5'] is None
        assert pc['l6'] is None

    def test_component_with_no_class_still_holds_a_slot(self, rows: dict[str, dict]) -> None:
        # component 2 has a class, so use the complex target to check both slots exist
        assert len(rows['CHEMBL_T3']['_metadata']['protein_classification']) == 2
```

- [ ] **Step 2: Run it to verify it fails**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -k Target -rxs`
Expected: FAIL with `PostgresExportError: no SQL file for query chembl_target`

- [ ] **Step 3: Write the query**

Create `pis/src/pis/sql/chembl_target.sql`:

```sql
-- Rebuild the chembl_<version>_target Elasticsearch document from the ChEMBL
-- relational schema.
--
-- Grain: one row per target_dictionary.tid.
--
-- `_metadata.protein_classification` MUST stay positionally aligned with
-- `target_components`: pts/src/pts/pyspark/target.py zips the two arrays. Both
-- lists are therefore built from the same component list, ordered by
-- targcomp_id, with exactly one classification slot per component.
WITH RECURSIVE ancestry AS (
    SELECT pc.protein_class_id AS leaf_id,
           pc.protein_class_id,
           pc.parent_id,
           pc.pref_name,
           pc.class_level
    FROM protein_classification pc
    UNION ALL
    SELECT a.leaf_id,
           pc.protein_class_id,
           pc.parent_id,
           pc.pref_name,
           pc.class_level
    FROM ancestry a
    JOIN protein_classification pc ON pc.protein_class_id = a.parent_id
),
levels AS (
    SELECT leaf_id,
           max(CASE WHEN class_level = 1 THEN pref_name END) AS l1,
           max(CASE WHEN class_level = 2 THEN pref_name END) AS l2,
           max(CASE WHEN class_level = 3 THEN pref_name END) AS l3,
           max(CASE WHEN class_level = 4 THEN pref_name END) AS l4,
           max(CASE WHEN class_level = 5 THEN pref_name END) AS l5,
           max(CASE WHEN class_level = 6 THEN pref_name END) AS l6
    FROM ancestry
    GROUP BY leaf_id
),
chosen_class AS (
    -- a component can carry several classes; the Elasticsearch document keeps
    -- one. Take the most specific, breaking ties on the lower id for stability
    SELECT component_id, protein_class_id
    FROM (
        SELECT cc.component_id,
               cc.protein_class_id,
               row_number() OVER (
                   PARTITION BY cc.component_id
                   ORDER BY pc.class_level DESC, cc.protein_class_id ASC
               ) AS rn
        FROM component_class cc
        JOIN protein_classification pc ON pc.protein_class_id = cc.protein_class_id
    )
    WHERE rn = 1
),
components AS (
    SELECT tc.tid,
           list(struct_pack(accession := cs.accession) ORDER BY tc.targcomp_id) AS target_components,
           list(struct_pack(
               protein_class_id := l.leaf_id,
               l1 := l.l1, l2 := l.l2, l3 := l.l3, l4 := l.l4, l5 := l.l5, l6 := l.l6
           ) ORDER BY tc.targcomp_id) AS protein_classification
    FROM target_components tc
    JOIN component_sequences cs ON cs.component_id = tc.component_id
    LEFT JOIN chosen_class ch ON ch.component_id = tc.component_id
    LEFT JOIN levels l ON l.leaf_id = ch.protein_class_id
    GROUP BY tc.tid
)
SELECT
    td.chembl_id AS target_chembl_id,
    td.pref_name,
    td.target_type,
    coalesce(c.target_components, []::STRUCT(accession VARCHAR)[]) AS target_components,
    struct_pack(
        protein_classification := coalesce(
            c.protein_classification,
            []::STRUCT(
                protein_class_id INTEGER,
                l1 VARCHAR, l2 VARCHAR, l3 VARCHAR, l4 VARCHAR, l5 VARCHAR, l6 VARCHAR
            )[]
        )
    ) AS _metadata
FROM target_dictionary td
LEFT JOIN components c ON c.tid = td.tid
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `uv run --frozen --directory pis pytest tests/test_chembl_queries.py -rxs`
Expected: PASS

- [ ] **Step 5: Compare against the real baseline and settle the class-selection rule**

```bash
uv run --frozen --directory pis python /tmp/chembl-baseline/compare_chembl_es.py \
  chembl_target <work_path>/input/drug/chembl_target.parquet \
  /tmp/chembl-baseline/chembl_target.jsonl
```

If `_metadata` mismatches, the harness prints a sample. Try the alternatives in this order: smallest `protein_class_id` regardless of level; the class whose `class_level` matches the component's depth in `component_class`; keeping every class and letting the array be longer. Iterate until `MATCH`, then update the query comment and the spec's open question 2.

Additionally, confirm the alignment invariant on the real data:

```bash
uv run --frozen --directory pis python -c "
import duckdb
print(duckdb.connect().execute('''
    SELECT count(*) FROM read_parquet('<work_path>/input/drug/chembl_target.parquet')
    WHERE len(target_components) <> len(_metadata.protein_classification)
''').fetchone())
"
```

Expected: `(0,)`.

- [ ] **Step 6: Lint, type-check and commit**

```bash
uv run --frozen --directory pis ruff check src tests
uv run --frozen --directory pis ty check src tests
git add pis/src/pis/sql/chembl_target.sql pis/tests/test_chembl_queries.py docs/superpowers/specs
git commit -m "pis: rebuild chembl targets from postgres"
```

---

### Task 9: Wire the queries into `pis/config.yaml` and remove the Elasticsearch tasks

**Files:**
- Modify: `pis/config.yaml:136-215` (drug/clinical_report step), `:950-965` (target step)

After this task `pis/src/pis/sql/` holds exactly the four real queries — the test fixture lives under `pis/tests/sql/` and never ships.

**Interfaces:**
- Consumes: everything from Tasks 1-8.
- Produces: the four parquet files at `input/drug/*.parquet`.

- [ ] **Step 1: Add the `queries:` block**

In `pis/config.yaml`, on the `postgres_export chembl tables` task added by PR #17, add after its `tables:` list:

```yaml
      queries:
        - query: chembl_molecule
          destination: input/drug/chembl_molecule.parquet
          requires_tables:
            - molecule_dictionary
            - compound_structures
            - molecule_hierarchy
            - molecule_synonyms
            - compound_records
            - source
        - query: chembl_mechanism
          destination: input/drug/chembl_mechanism.parquet
          requires_tables:
            - drug_mechanism
            - mechanism_refs
            - molecule_dictionary
            - molecule_hierarchy
            - target_dictionary
        - query: chembl_target
          destination: input/drug/chembl_target.parquet
          requires_tables:
            - target_dictionary
            - target_components
            - component_sequences
            - component_class
            - protein_classification
        - query: chembl_drug_warning
          destination: input/drug/chembl_drug_warning.parquet
          requires_tables:
            - drug_warning
            - warning_refs
            - molecule_dictionary
            - molecule_hierarchy
```

- [ ] **Step 2: Remove the four Elasticsearch tasks in the drug step**

Delete the four tasks named `elasticsearch chembl drug warning`, `elasticsearch chembl mechanism of action`, `elasticsearch chembl molecule` and `elasticsearch chembl target` (currently `pis/config.yaml:260-316`).

- [ ] **Step 3: Remove the Elasticsearch task in the target step**

Delete the `elasticsearch chembl target` task at `pis/config.yaml:954-963`, which wrote `input/target/chembl/chembl_target.jsonl`. PTS is repointed at `input/drug/chembl_target.parquet` in Task 10.

- [ ] **Step 4: Verify the config still parses and no Elasticsearch task survives**

Run: `uv run --frozen --directory pis pis --help`
Expected: exits 0.

Run:

```bash
uv run --frozen --directory pis python -c "
import yaml
config = yaml.safe_load(open('config.yaml'))
names = [t['name'] for step in config['steps'].values() for t in step]
survivors = [n for n in names if n.startswith('elasticsearch')]
assert not survivors, survivors
queries = [
    q['query']
    for step in config['steps'].values()
    for t in step
    for q in t.get('queries', [])
]
assert sorted(queries) == [
    'chembl_drug_warning', 'chembl_mechanism', 'chembl_molecule', 'chembl_target'
], queries
print(f'{len(names)} tasks, no elasticsearch, 4 queries wired')
"
```

Expected: prints the summary line, no assertion error.

- [ ] **Step 5: Confirm no reference to the removed outputs remains in PIS**

Run: `grep -rn "chembl_molecule.jsonl\|chembl_mechanism.jsonl\|chembl_target.jsonl\|chembl_drug_warning.jsonl" pis/`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add pis/config.yaml
git commit -m "pis: read chembl drug data from postgres instead of elasticsearch"
```

---

### Task 10: Repoint PTS at the parquet files

**Files:**
- Modify: `pts/config.yaml:405`, `:663`, `:672-673`, `:683`
- Modify: `pts/src/pts/pyspark/drug_warning.py:32`
- Modify: `pts/src/pts/pyspark/drug_mechanism_of_action.py:36-37`
- Modify: `pts/src/pts/pyspark/chembl_molecule.py` (the molecule load)
- Modify: `pts/src/pts/pyspark/target.py:146`

**Interfaces:**
- Consumes: the four parquet files produced in Task 9.
- Produces: no new interfaces. Column names and nesting are unchanged, so no logic changes.

- [ ] **Step 1: Update the PTS config sources**

In `pts/config.yaml` make these four edits:

- `:405` `chembl: input/target/chembl/chembl_target.jsonl` → `chembl: input/drug/chembl_target.parquet`
- `:663` `source: input/drug/chembl_drug_warning.jsonl` → `source: input/drug/chembl_drug_warning.parquet`
- `:672` `chembl_mechanism: input/drug/chembl_mechanism.jsonl` → `.parquet`
- `:673` `chembl_target: input/drug/chembl_target.jsonl` → `.parquet`
- `:683` `chembl_molecule: input/drug/chembl_molecule.jsonl` → `.parquet`

- [ ] **Step 2: Update the readers**

`pts/src/pts/pyspark/drug_warning.py:32`:

```python
    warnings_df = spark.load_data(source)
```

`pts/src/pts/pyspark/drug_mechanism_of_action.py:36-37`:

```python
    mechanism_df = spark.load_data(source['chembl_mechanism'])
    target_df = spark.load_data(source['chembl_target'])
```

`pts/src/pts/pyspark/chembl_molecule.py`, the molecule load — drop the `format='json'` argument so it matches the others.

`pts/src/pts/pyspark/target.py:146`:

```python
    chembl_raw = spark.read.parquet(source['chembl'])
```

Dropping the argument is correct: `Session.load_data` at `pts/src/pts/pyspark/common/session.py:75` is declared `format: str = 'parquet'`, so parquet is already the default. Do not add `format='parquet'` explicitly — it would be redundant with every other call site in the package.

- [ ] **Step 3: Run the PTS test suite**

Run: `uv run --frozen --directory pts pytest -rxs`
Expected: PASS. These modules have no unit tests that read the files, so this is a regression check on everything else.

- [ ] **Step 4: Lint and type-check PTS**

Run: `uv run --frozen --directory pts ruff check src tests`
Run: `uv run --frozen --directory pts ty check src tests`
Expected: `All checks passed!` from both.

- [ ] **Step 5: Confirm no JSONL reference remains**

Run: `grep -rn "chembl_molecule.jsonl\|chembl_mechanism.jsonl\|chembl_target.jsonl\|chembl_drug_warning.jsonl" pts/`
Expected: no output.

- [ ] **Step 6: Commit**

```bash
git add pts/config.yaml pts/src/pts/pyspark
git commit -m "pts: read chembl drug data from parquet"
```

- [ ] **Step 7: Run the full check across both packages**

Run: `make lint`
Run: `make test`
Expected: both pass.

---

## Verification evidence for the PR description

Collect and paste into the PR body:

- The four `compare_chembl_es.py` runs, each showing `MATCH` with its row counts.
- The answers found for the two open questions: how `cross_references` is really built, and which protein class the Elasticsearch document keeps.
- The alignment check from Task 8 Step 5 returning `(0,)`.
- Output of `make lint` and `make test`.
