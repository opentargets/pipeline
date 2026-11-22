"""scripts/schema_drift.py: disagreements between the dataset models and the ClickHouse landing DDL."""

import importlib.util
import sys
from pathlib import Path
from typing import Annotated, Literal

import pytest

pytest.importorskip('ot_datamodels')

import polars as pl
from ot_datamodels.base import ClickHouseType, DatasetModel, PolarsDtype, StructModel

ROOT = Path(__file__).parents[1]


def _load_script():
    spec = importlib.util.spec_from_file_location('schema_drift', ROOT / 'scripts' / 'schema_drift.py')
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules['schema_drift'] = module
    spec.loader.exec_module(module)
    return module


schema_drift = _load_script()


class Info(StructModel):
    a: str
    b: int
    c: str


class Alpha(DatasetModel):
    dataset_name = 'alpha'
    dataset_title = 'Alpha'
    dataset_description = 'Alpha.'
    example = {}

    id: str
    extra: str
    count: Annotated[int, PolarsDtype(pl.Int32)]
    flag: str
    chrom: str
    biotype: str
    marked: Annotated[int, ClickHouseType('UInt8')]
    ok: bool
    tags: list[str] | None
    info: Info | None
    level: Literal['x', 'y']


class EvidenceA(DatasetModel):
    dataset_name = 'evidence_a'
    dataset_title = 'A'
    dataset_description = 'A.'
    example = {}

    datasourceId: str  # noqa: N815
    score: float | None


class EvidenceB(DatasetModel):
    dataset_name = 'evidence_b'
    dataset_title = 'B'
    dataset_description = 'B.'
    example = {}

    datasourceId: str  # noqa: N815
    score: float


MODELS = {m.dataset_name: m for m in (Alpha, EvidenceA, EvidenceB)}

DATASETS_YAML = """
clickhouse:
  alpha:
    table: alpha_log
    input_dir: output/alpha
    model: alpha
    preload_script: config/clickhouse/schema/alpha.sql
    postload_script: config/clickhouse/scripts/alpha.sql
  evidence:
    table: evidence_log
    input_dir: output
    models: 'evidence_*'
    preload_script: config/clickhouse/schema/evidence.sql
  generated:
    table: generated_log
    input_dir: output/alpha
    model: alpha
  viewed:
    table: viewed_log
    input_dir: view/thing
    preload_script: config/clickhouse/schema/viewed.sql
  built:
    table: built
    derived: true
    preload_script: config/clickhouse/scripts/built.sql
opensearch:
  ignored: {}
"""

ALPHA_DDL = """-- landing table
CREATE TABLE if not exists alpha_log (
    `count` UInt32,
    `id` String,
    `flag` Float64,
    chrom Enum('1', 'X'),
    `biotype` LowCardinality (String),
    `marked` Nullable (UInt8),
    `ok` Boolean,
    `tags` Array(String),
    `info` Tuple(b Int64, `a` String),
    `level` String,
    `legacy` String,
) engine = Log;
"""

EVIDENCE_DDL = """CREATE TABLE if not exists evidence_log (
  `datasourceId` String,
  `score` Nullable(Float64),
  `old` Nullable(String)
) ENGINE = Log;
"""

ALPHA_SCRIPT = """CREATE TABLE alpha ENGINE = Log AS (
    SELECT *
    FROM alpha_log
);
SELECT max(count), any(extra) FROM alpha;
"""

# reads two other datasets' landing tables: evidence through an alias, alpha by column name
OTHER_SCRIPT = """CREATE TABLE other ENGINE = Log AS (
    SELECT e.*, flag
    FROM evidence_log AS e
    LEFT JOIN alpha_log ON alpha_log.id = e.datasourceId
);
"""

# only a table whose name starts with the landing name: not a read of alpha_log
MERGETREE_SCRIPT = """SELECT * FROM alpha_log_mergetree;
"""

COLUMNS = (
    'category',
    'dataset',
    'model',
    'field',
    'model_type',
    'model_clickhouse',
    'ddl_type',
    'model_nullable',
    'ddl_nullable',
    'serving_read',
)

PASS = 'pass:alpha.sql'  # noqa: S105
OTHER = 'pass:other.sql'
EXPECTED = [
    ('model_only', 'alpha', 'alpha', 'extra', 'String', '', '', 'false', '', 'no'),
    ('type', 'alpha', 'alpha', 'count', 'Int32', '', 'UInt32', 'false', 'false', 'named:alpha.sql'),
    ('kind', 'alpha', 'alpha', 'flag', 'String', '', 'Float64', 'false', 'false', 'named:other.sql;pass:alpha.sql'),
    ('open_vs_enum', 'alpha', 'alpha', 'chrom', 'String', '', "Enum('1', 'X')", 'false', 'false', PASS),
    ('low_cardinality', 'alpha', 'alpha', 'biotype', 'String', '', 'LowCardinality(String)', 'false', 'false', PASS),
    ('nullability', 'alpha', 'alpha', 'marked', 'Int64', 'UInt8', 'Nullable(UInt8)', 'false', 'true', PASS),
    ('list_nullability', 'alpha', 'alpha', 'tags', 'Array(String)', '', 'Array(String)', 'true', 'false', PASS),
    (
        'struct_nullability',
        'alpha',
        'alpha',
        'info',
        'Tuple(a String, b Int64, c String)',
        '',
        'Tuple(b Int64, a String)',
        'true',
        'false',
        PASS,
    ),
    ('struct_order', 'alpha', 'alpha', 'info', 'a, b', '', 'b, a', '', '', PASS),
    ('struct_field_missing', 'alpha', 'alpha', 'info/c', 'String', '', '', 'false', '', 'no'),
    ('literal_vs_string', 'alpha', 'alpha', 'level', 'String', '', 'String', 'false', 'false', PASS),
    ('ddl_only', 'alpha', 'alpha', 'legacy', '', '', 'String', '', 'false', PASS),
    (
        'column_order',
        'alpha',
        'alpha',
        '',
        'id, count, flag, chrom, biotype, marked, ok, tags, info, level',
        '',
        'count, id, flag, chrom, biotype, marked, ok, tags, info, level',
        '',
        '',
        PASS,
    ),
    ('ddl_only', 'evidence', '', 'old', '', '', 'Nullable(String)', '', 'true', OTHER),
    ('nullability', 'evidence', 'evidence_b', 'score', 'Float64', '', 'Nullable(Float64)', 'false', 'true', OTHER),
    ('no_model', 'viewed', '', '', '', '', '', '', '', 'no'),
]


@pytest.fixture
def config(tmp_path: Path) -> tuple[Path, Path, Path]:
    schema_dir = tmp_path / 'schema'
    scripts_dir = tmp_path / 'scripts'
    schema_dir.mkdir()
    scripts_dir.mkdir()
    datasets_yaml = tmp_path / 'datasets.yaml'
    datasets_yaml.write_text(DATASETS_YAML)
    (schema_dir / 'alpha.sql').write_text(ALPHA_DDL)
    (schema_dir / 'evidence.sql').write_text(EVIDENCE_DDL)
    (scripts_dir / 'alpha.sql').write_text(ALPHA_SCRIPT)
    (scripts_dir / 'other.sql').write_text(OTHER_SCRIPT)
    (scripts_dir / 'mergetree.sql').write_text(MERGETREE_SCRIPT)
    return datasets_yaml, schema_dir, scripts_dir


def _key(row: tuple[str, ...]) -> tuple[str, ...]:
    return row[1], row[2], row[3], row[0]


def test_drift_lists_one_row_per_disagreement(config):
    frame = schema_drift.drift(*config, models=MODELS)
    assert frame.columns == list(COLUMNS)
    assert sorted(frame.rows(), key=_key) == sorted(EXPECTED, key=_key)


def test_drift_covers_every_category(config):
    frame = schema_drift.drift(*config, models=MODELS)
    assert set(frame['category']) == set(schema_drift.CATEGORIES) | {'no_model'}


def test_marker_equal_to_ddl_is_no_type_disagreement(config):
    frame = schema_drift.drift(*config, models=MODELS)
    marked = frame.filter(pl.col('field') == 'marked')
    assert marked['category'].to_list() == ['nullability']


@pytest.mark.parametrize(
    ('ddl', 'rendered'),
    [
        ('Nullable (LowCardinality (String))', 'Nullable(LowCardinality(String))'),
        ('LowCardinality(Nullable(String))', 'Nullable(LowCardinality(String))'),
        ("Enum8 ('-1' = -1, '0' = 0, '1' = 1)", "Enum8('-1' = -1, '0' = 0, '1' = 1)"),
        ('Array (Nullable (String))', 'Array(Nullable(String))'),
        ('Array(Tuple(`id` String, source Nullable(String),))', 'Array(Tuple(id String, source Nullable(String)))'),
        ('FixedString (2)', 'FixedString(2)'),
        ('Boolean', 'Bool'),
    ],
)
def test_parse_type_normalises(ddl, rendered):
    assert schema_drift.render(schema_drift.parse_type(ddl)) == rendered


def test_unlanded_fields_are_never_read(config):
    frame = schema_drift.drift(*config, models=MODELS)
    unlanded = frame.filter(pl.col('category').is_in(['model_only', 'struct_field_missing']))
    assert 'extra' in unlanded['field'].to_list()  # named in alpha.sql, but never loaded
    assert set(unlanded['serving_read']) == {'no'}


@pytest.mark.parametrize(
    ('sql', 'reads'),
    [
        ('SELECT * FROM t_log', True),
        ('select *\n  from\n    t_log\n', True),
        ('SELECT * EXCEPT (a, b) FROM t_log', True),
        ('SELECT * EXCEPT a FROM t_log', True),
        ('SELECT rowNumberInAllBlocks() AS rowId,\n *\n FROM t_log', True),
        ('SELECT *, x FROM t_log', True),
        ('SELECT count(*) FROM t_log', False),
        ('SELECT * FROM (SELECT a FROM t_log)', False),
        ('SELECT t_log.* FROM t_log', True),
        ('SELECT i.* FROM t_log AS i', True),
        ('SELECT i.* FROM other LEFT JOIN t_log i ON true', True),
        ('SELECT * FROM t_log_mergetree', False),
        ('SELECT * FROM t_logx', False),
        ('SELECT x.* FROM t_log AS i JOIN other AS x ON true', False),
        ('SELECT i.* FROM t_log LEFT JOIN other AS i ON true', False),
        ('SELECT a FROM t_log', False),
    ],
)
def test_reads_star(sql, reads):
    assert schema_drift._reads_star(sql, 't_log') is reads


@pytest.mark.parametrize(
    ('path', 'read'),
    [
        ('info[]/a', 'named:s.sql'),  # its column is named
        ('other/b', 'no'),
        ('', 'no'),
    ],
)
def test_nested_field_is_named_through_its_column(path, read):
    serving = schema_drift._Serving('t_log', (('s.sql', 'SELECT info FROM t_log'),))
    assert serving.read(path, landed=True) == read


def test_parse_table_reads_columns_in_order():
    columns = schema_drift.parse_table(ALPHA_DDL)
    assert list(columns)[:3] == ['count', 'id', 'flag']
    assert list(columns)[-1] == 'legacy'


def test_real_config_has_every_landing_table():
    frame = schema_drift.drift(
        ROOT / 'config' / 'datasets.yaml',
        ROOT / 'config' / 'clickhouse' / 'schema',
        ROOT / 'config' / 'clickhouse' / 'scripts',
    )
    assert set(frame['category']) <= set(schema_drift.CATEGORIES) | {'no_model'}
    assert set(frame.filter(pl.col('category') == 'no_model')['dataset']) == {
        'aotf',
        'association_time_series',
        'protein_coding_coords',
        'targets',
    }
