"""Shared I/O, discovery, and validation helpers for LDSC-CTS tasks."""

from __future__ import annotations

import json
import math
import re
import shlex
from collections.abc import Mapping, Sequence
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


def join_input_path(root: str, path: str) -> str:
    """Resolve a registry path against an input root."""
    value = str(path).strip()
    if not value:
        raise ValueError('Input path must not be empty')
    if value.startswith('/') or '://' in value:
        return value
    return f'{str(root).rstrip("/")}/{value.lstrip("/")}'


def normalise_dataset_registry(raw: Any, key: str = 'datasets') -> list[dict[str, Any]]:
    """Validate and normalise a list of ``{id, path, ...}`` dataset settings."""
    if not isinstance(raw, Sequence) or isinstance(raw, (str, bytes, Mapping)):
        raise TypeError(f'{key} must be a list of dataset objects')
    datasets: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, value in enumerate(raw):
        if not isinstance(value, Mapping):
            raise TypeError(f'{key}[{index}] must be an object')
        dataset = dict(value)
        dataset_id = '' if dataset.get('id') is None else str(dataset.get('id')).strip()
        path = '' if dataset.get('path') is None else str(dataset.get('path')).strip()
        if not dataset_id:
            raise ValueError(f'{key}[{index}] is missing a non-empty id')
        if dataset_id in seen:
            raise ValueError(f"{key} contains duplicate id '{dataset_id}'")
        if not path:
            raise ValueError(f"{key}[{index}] '{dataset_id}' is missing a non-empty path")
        seen.add(dataset_id)
        dataset['id'] = dataset_id
        dataset['path'] = path
        datasets.append(dataset)
    if not datasets:
        raise ValueError(f'{key} must contain at least one dataset')
    return datasets


def normalise_populations(raw: Any, key: str = 'populations') -> list[str]:
    """Validate a non-empty, duplicate-free list of LD population labels."""
    if isinstance(raw, str):
        values: list[Any] = raw.split(',')
    elif isinstance(raw, Sequence) and not isinstance(raw, (bytes, Mapping)):
        values = list(raw)
    else:
        raise TypeError(f'{key} must be a list of population labels')
    populations: list[str] = []
    seen: set[str] = set()
    for value in values:
        population = '' if value is None else str(value).strip().lower()
        if not population:
            raise ValueError(f'{key} contains an empty population label')
        if population in seen:
            raise ValueError(f"{key} contains duplicate population '{population}'")
        seen.add(population)
        populations.append(population)
    if not populations:
        raise ValueError(f'{key} must contain at least one population')
    return populations


def normalise_chromosome(chromosome: int | str) -> str:
    """Normalise ``chrN`` and ``N`` to the canonical variant-id chromosome."""
    value = str(chromosome).strip()
    if value.lower().startswith('chr'):
        value = value[3:]
    if not value.isdigit() or not 1 <= int(value) <= 22:
        raise ValueError(f'chromosome must be an autosome from 1 through 22: {chromosome}')
    return str(int(value))


def normalise_chromosomes(raw: Any) -> list[str]:
    """Normalise an optional chromosome setting to unique autosome labels."""
    values = range(1, 23) if raw is None else raw
    if isinstance(values, int):
        values = [values]
    elif isinstance(values, bytes):
        values = values.decode().split(',')
    elif isinstance(values, str):
        values = values.split(',')
    chromosomes = list(dict.fromkeys(normalise_chromosome(value) for value in values))
    if not chromosomes:
        raise ValueError('chromosomes must contain at least one autosome')
    return chromosomes


def reference_path(
    root: str,
    population: str,
    settings: Mapping[str, Any],
    setting_name: str,
    default_filename: str,
) -> str:
    """Resolve a population-specific reference path from shared settings."""
    configured = settings.get(f'{setting_name}_paths', {})
    if isinstance(configured, Mapping) and population in configured:
        value = configured[population]
        if value is None or not str(value).strip():
            raise ValueError(f'{setting_name}_paths[{population}] must not be empty')
        return join_input_path(root, str(value))
    template = settings.get(f'{setting_name}_template')
    filename = str(template).format(population=population) if template else default_filename
    return join_input_path(join_input_path(root, population), filename)


def _hadoop_filesystem(spark: SparkSession, path: str):
    """Return the Hadoop path and filesystem for local or GCS paths."""
    jvm = spark._jvm
    jsc = spark._jsc
    if jvm is None or jsc is None:
        raise RuntimeError('Spark JVM is unavailable while inspecting a filesystem path')
    hadoop_path = jvm.org.apache.hadoop.fs.Path(path)
    return hadoop_path, hadoop_path.getFileSystem(jsc.hadoopConfiguration())


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
        if not isinstance(value, str) or not value:
            raise TypeError(f'Edge manifest path for {ancestry}/chr{chromosome} must be a string')
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
    try:
        min_r2 = float(values['--min-r2'])
        window_cm = float(values['--ld-window-cm'])
    except ValueError as exc:
        raise ValueError(f'Invalid edge-window arguments: {command}') from exc
    if min_r2 != 0.0:
        raise ValueError('LDSC-CTS edge reference must use --min-r2 0.0')
    if window_cm != 1.0:
        raise ValueError('LDSC-CTS edge reference must use --ld-window-cm 1.0')
    if values['--output-path'] != 'edges.parquet':
        raise ValueError('Unexpected edge output path in edge exporter command')
    return {'ancestry': match.group(1).lower(), 'chromosome': chromosome}


def success_exists(spark: SparkSession, path: str) -> bool:
    """Check a Spark output's ``_SUCCESS`` marker on local or GCS storage."""
    if '://' not in path:
        return (Path(path) / '_SUCCESS').is_file()
    hadoop_path, fs = _hadoop_filesystem(spark, f'{path.rstrip("/")}/_SUCCESS')
    return bool(fs.exists(hadoop_path))


def discover_completed_study_paths(
    spark: SparkSession,
    root: str,
    study_ids: Sequence[str] | None = None,
) -> list[tuple[str | None, str]]:
    """Discover completed one-study datasets beneath a local or GCS root.

    The normal layout is ``root/<studyId>/_SUCCESS``. A root which is itself a
    completed dataset is also accepted when exactly one ``study_ids`` filter is
    supplied; this keeps small local fixtures convenient without weakening the
    production one-study-per-child invariant.
    """
    root = root.rstrip('/')
    requested = [str(value).strip() for value in study_ids or [] if str(value).strip()]
    if study_ids is not None and not requested:
        raise ValueError('study_ids must contain at least one non-empty study ID when supplied')

    if requested:
        entries: list[tuple[str | None, str]] = []
        missing: list[str] = []
        for study_id in requested:
            path = f'{root}/{study_id}'
            if success_exists(spark, path):
                entries.append((study_id, path))
            elif len(requested) == 1 and success_exists(spark, root):
                entries.append((study_id, root))
            else:
                missing.append(study_id)
        if missing:
            raise ValueError(f'Missing completed summary-statistics datasets: {missing}')
        return entries

    if '://' not in root:
        root_path = Path(root)
        if not root_path.is_dir():
            raise FileNotFoundError(f'Summary-statistics root does not exist: {root}')
        children = [(child.name, str(child)) for child in sorted(root_path.iterdir()) if child.is_dir()]
        entries = [
            (name, path)
            for name, path in children
            if not name.startswith('_') and success_exists(spark, path)
        ]
    else:
        hadoop_root, fs = _hadoop_filesystem(spark, root)
        if not fs.exists(hadoop_root):
            raise FileNotFoundError(f'Summary-statistics root does not exist: {root}')
        entries = []
        for status in fs.listStatus(hadoop_root):
            if not status.isDirectory():
                continue
            child = status.getPath()
            name = str(child.getName())
            path = str(child)
            if not name.startswith('_') and success_exists(spark, path):
                entries.append((name, path))
        entries.sort(key=lambda item: item[0])

    if not entries and success_exists(spark, root):
        return [(None, root)]
    if not entries:
        raise ValueError(f'No completed summary-statistics datasets found below {root}')
    return entries


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
