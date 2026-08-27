"""Shared I/O and validation helpers for the PTS LDSC-CTS tasks."""

from __future__ import annotations

import json
import math
import re
import shlex
from pathlib import Path
from typing import Any

import pyspark.sql.functions as f
from pyspark.sql import DataFrame, SparkSession

_ANCESTRY_RE = re.compile(r'\.([a-z0-9]+)\.common\.adj\.ld\.bm$')
_REQUIRED_EDGE_OPTIONS = (
    '--ld-bm-path',
    '--chromosome',
    '--min-r2',
    '--ld-window-cm',
    '--output-path',
)


def quoted_col(name: str):
    """Return a Spark column reference for a literal physical column name."""
    return f.col(f"`{name.replace('`', '``')}`")


def read_table(spark: SparkSession, path: str, fmt: str = 'parquet', sep: str = '\t') -> DataFrame:
    """Read a parquet or delimited table with a small, explicit interface."""
    fmt = fmt.lower()
    if fmt == 'parquet':
        return spark.read.parquet(path)
    if fmt in {'csv', 'tsv'}:
        if sep == r'\t':
            sep = '\t'
        return spark.read.csv(path, header=True, sep=sep, inferSchema=True)
    raise ValueError(f"Unsupported table format '{fmt}'")


def normalise_chromosome(chromosome: int | str) -> str:
    """Normalise ``chrN`` and ``N`` to the canonical variant-id chromosome."""
    value = str(chromosome).strip()
    if value.lower().startswith('chr'):
        value = value[3:]
    if not value.isdigit() or not 1 <= int(value) <= 22:
        raise ValueError(f'chromosome must be an autosome from 1 through 22: {chromosome}')
    return str(int(value))


def discover_edge_manifest(
    work_root: str,
    ancestry: str,
    chromosomes: list[int | str],
) -> dict[str, str]:
    """Find one completed, compatible Nextflow edge export per chromosome."""
    root = Path(work_root)
    if not root.is_dir():
        raise FileNotFoundError(f'LD edge work root does not exist: {work_root}')

    requested = [normalise_chromosome(c) for c in chromosomes]
    manifest: dict[str, str] = {}
    ancestry = ancestry.lower()
    for success in root.rglob('edges.parquet/_SUCCESS'):
        edge_dir = success.parent
        command_path = edge_dir.parent / '.command.sh'
        if not command_path.is_file():
            raise ValueError(f'Completed edge dataset has no command file: {edge_dir}')
        parsed = _parse_edge_command(command_path.read_text())
        if parsed['ancestry'] != ancestry or parsed['chromosome'] not in requested:
            continue
        chromosome = parsed['chromosome']
        if chromosome in manifest:
            raise ValueError(
                f'Duplicate successful edge datasets for {ancestry}/chr{chromosome}: '
                f'{manifest[chromosome]} and {edge_dir}'
            )
        manifest[chromosome] = str(edge_dir)

    missing = [chromosome for chromosome in requested if chromosome not in manifest]
    if missing:
        raise ValueError(f'Missing successful edge datasets for {ancestry}: {missing}')
    return {chromosome: manifest[chromosome] for chromosome in requested}


def load_edge_manifest(path: str, ancestry: str, chromosomes: list[int | str]) -> dict[str, str]:
    """Load an explicit JSON edge manifest, validating requested entries."""
    raw = json.loads(Path(path).read_text())
    if ancestry not in raw:
        raise ValueError(f"Edge manifest has no ancestry '{ancestry}'")
    requested = [normalise_chromosome(c) for c in chromosomes]
    values = raw[ancestry]
    if not isinstance(values, dict):
        raise TypeError(f"Edge manifest ancestry '{ancestry}' must be an object")
    result = {}
    for chromosome in requested:
        value = values.get(f'chr{chromosome}') or values.get(chromosome)
        if value is None:
            raise ValueError(f'Edge manifest is missing {ancestry}/chr{chromosome}')
        result[chromosome] = value
    return result


def _parse_edge_command(command: str) -> dict[str, str]:
    """Parse and validate exporter arguments recorded by Nextflow."""
    tokens = shlex.split(command)
    values: dict[str, str] = {}
    for option in _REQUIRED_EDGE_OPTIONS:
        try:
            values[option] = tokens[tokens.index(option) + 1]
        except (ValueError, IndexError) as exc:
            raise ValueError(f'Edge command is missing {option}: {command}') from exc

    match = _ANCESTRY_RE.search(values['--ld-bm-path'])
    if match is None:
        raise ValueError(f'Could not infer ancestry from exporter command: {command}')
    chromosome = normalise_chromosome(values['--chromosome'])
    if float(values['--min-r2']) != 0.0:
        raise ValueError('LDSC-CTS edge reference must use --min-r2 0.0')
    if float(values['--ld-window-cm']) != 1.0:
        raise ValueError('LDSC-CTS edge reference must use --ld-window-cm 1.0')
    if values['--output-path'] != 'edges.parquet':
        raise ValueError('Unexpected edge output path in edge exporter command')
    return {'ancestry': match.group(1), 'chromosome': chromosome}


def success_exists(spark: SparkSession, path: str) -> bool:
    """Check a Spark output's ``_SUCCESS`` marker on local or GCS storage."""
    if not path.startswith('gs://'):
        return (Path(path) / '_SUCCESS').is_file()
    jvm = spark._jvm
    jsc = spark._jsc
    if jvm is None or jsc is None:
        raise RuntimeError('Spark JVM is unavailable while checking output completion')
    hadoop_path = jvm.org.apache.hadoop.fs.Path(f'{path.rstrip("/")}/_SUCCESS')
    fs = hadoop_path.getFileSystem(jsc.hadoopConfiguration())
    return bool(fs.exists(hadoop_path))


def variant_positions_from_edges(edges: DataFrame) -> DataFrame:
    """Parse chromosome and position from both sides of an edge list."""
    universe = (
        edges.select('variantId')
        .union(edges.select(f.col('tagVariantId').alias('variantId')))
        .distinct()
    )
    parts = f.split(f.col('variantId'), '_')
    return universe.select(
        'variantId',
        parts.getItem(0).alias('chromosome'),
        parts.getItem(1).cast('long').alias('position'),
    ).filter(f.col('position').isNotNull())


def finite_or_none(value: Any) -> float | None:
    """Convert numeric output to a nullable finite float."""
    if value is None:
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None
