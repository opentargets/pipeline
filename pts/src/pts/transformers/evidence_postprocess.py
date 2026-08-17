"""Validate and post-process Open Targets Platform evidence.

Polars entry point for the `evidence_postprocess_*` config.yaml steps, replacing
`pts.pyspark.evidence_postprocess`. It turns otter's `source`/`destination`/`settings` into the
parameters the recipe needs, reads, runs `pts.transformers.evidence.postprocess`, and writes. The
processing itself lives in the `pts.transformers.evidence` package.

It stays here rather than inside that package because otter resolves a transformer by importing
`pts.transformers.<name>` and taking the attribute of the same name, so a step module cannot live
in a subpackage without changing that loader.

Reading lives here too, rather than in the package, because it is driven by `evidence_format` --
a settings key, so translating it is this module's job. The recipe takes a frame and knows nothing
about storage, so a future per-datasource module inherits no reading from it: one generating its
evidence in polars reads nothing, and one generating in spark reads its own parquet back through
`scan_dataset` directly, with no format dispatch to reuse. Should such a module ever own one of the
json sources, `_json_schema` is what it would want, and extracting it then -- with a real second
caller to shape it -- beats guessing at the boundary now.

The registry lookup happens HERE, not in the recipe. `EXPRESSIONS` is keyed by `datasource_id` --
NOT taken from `settings['score_expression']` / `settings['direction_on_*_expression']`, even though
config.yaml still carries those spark-SQL strings for every `evidence_postprocess_*` step. The
compiler that used to translate them (`pts.transformers.utils.spark_sql`) has been deleted; the
strings are inert leftovers until a later change strips them from config.yaml for every datasource
at once. As per-datasource modules take over, each will supply its own expressions and this lookup
shrinks with the registry.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config
from otter.storage.synchronous.handle import StorageHandle

from pts.schemas.evidence import evidence_schema
from pts.transformers.evidence.expressions import EXPRESSIONS
from pts.transformers.evidence.postprocess import EvidencePostprocessor, ValidationLuts
from pts.transformers.utils.dataset import scan_dataset, write_dataset
from pts.transformers.utils.validation_lut import build_disease_lut, build_publication_lut, build_target_lut


def _json_schema(source: str) -> dict[str, Any]:
    """The schema to pin for one json evidence part: every column it actually carries, typed.

    A full `schema=evidence_schema` pin does not just constrain dtypes, it MATERIALISES every one
    of the schema's 109 fields whether or not the source carries it -- measured:
    `reactome.json.gz`'s 12 real columns became 109 that way, filled with ~97 spurious all-null
    columns spark never produced. Spark's own json reader infers only the columns actually present,
    and harmonisation only casts columns that are in both the frame and the target schema, so a
    wholesale materialise-every-field pin has no spark equivalent to justify it.

    The columns actually present are discovered with a FULL-file scan (`infer_schema_length=None`),
    not a bounded sample: a bounded 5,000-row sample measurably missed one of `cosmic.json.gz`'s
    real 9 columns in a real run here. Each discovered column is then typed with `evidence_schema`'s
    dtype where the column is one of its fields -- so casting downstream still lands on that type --
    or the inferred dtype otherwise, mapping an inferred Null (an all-null column with no
    `evidence_schema` field to borrow a type from) to String.

    Deliberately not `scan_dataset`: that reader forwards no parsing options to ndjson, and this
    needs `infer_schema_length=None`. The measurement above is the whole point of this function, so
    it reads directly rather than losing the full-file scan to a shared reader that cannot express
    it.

    Args:
        source: one json evidence part's path (plain or gzip -- polars decompresses gzip natively
            from a path, which is also what keeps the read lazy).

    Returns:
        `{column: dtype}` for exactly the columns `source` carries, in ALPHABETICAL order.
    """
    inferred = pl.scan_ndjson(source, infer_schema_length=None).collect_schema()
    # Sorted by name, not `inferred`'s own order: passing this dict as a schema also fixes the
    # resulting frame's COLUMN ORDER, not just its dtypes -- and that order must be alphabetical to
    # match spark. Spark's json reader sorts inferred columns alphabetically; polars' keeps the
    # order columns first appear in the file. Measured on a one-line
    # `{"zebra":..,"alpha":..,"mango":..}` fixture: spark yields `['alpha', 'mango', 'zebra']`,
    # polars (pre-sort) yields `['zebra', 'alpha', 'mango']`. Left unsorted, that source order
    # propagates through harmonisation all the way to `validate_uniqueness`'s content hash, which
    # iterates the frame's columns in whatever order they are in -- an unsorted reader silently
    # picks a different surviving row among same-`id` duplicates than spark did.
    return {
        name: evidence_schema.get(name, inferred[name] if inferred[name] != pl.Null else pl.String)
        for name in sorted(inferred)
    }


def _read_evidence(path: str, evidence_format: str) -> pl.LazyFrame:
    """Read one datasource's raw evidence, before harmonisation to `evidence_schema`.

    Args:
        path: location of the evidence -- a directory of parquet parts, a single parquet file, or
            a single (possibly gzip-compressed) ndjson file.
        evidence_format: `settings['evidence_format']`, `'parquet'` or `'json'`.

    Returns:
        LazyFrame of the raw evidence, in its source columns/dtypes.

    Raises:
        ValueError: if `evidence_format` is neither `'parquet'` nor `'json'` -- only those two
            occur in config.yaml today, so an unrecognised value is a config error, not a case to
            fall through to the json reader silently. Also raised if `evidence_format` is `'json'`
            and `path` is a directory: every json evidence source in config.yaml is a single file,
            and the schema pin above measures ONE file, so a directory here is a config error
            refused up front rather than left to fail confusingly inside the schema scan.
    """
    if evidence_format == 'parquet':
        return scan_dataset(path)
    if evidence_format == 'json':
        if StorageHandle(path).stat().is_dir:
            msg = f'json evidence must be a single file, got a directory: {path!r}'
            raise ValueError(msg)
        return scan_dataset(path, format='ndjson', schema=_json_schema(path))
    msg = f'unrecognised evidence_format {evidence_format!r} for {path!r}, expected "parquet" or "json"'
    raise ValueError(msg)


def evidence_postprocess(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Harmonise, validate, date and score one datasource's evidence, splitting valid from failed.

    Args:
        source: `evidence_path`, `target_path`, `disease_path`, `publication_date_lut`.
        destination: `evidence` and `failed_evidence` output directories.
        settings: `datasource_id`, `evidence_format`, `unique_fields`, and optionally
            `excluded_biotypes` -- see the module docstring for why the scoring/direction
            expressions come from `EXPRESSIONS`, not `settings`.
        config: otter config; unused, required by the `Transform` task's transformer signature.

    Raises:
        KeyError: if `datasource_id` has no entry in `EXPRESSIONS`.
    """
    datasource_id = settings['datasource_id']
    logger.info(f'processing "{datasource_id}" evidence')
    try:
        expressions = EXPRESSIONS[datasource_id]
    except KeyError:
        msg = f'no score/direction expressions registered for datasource {datasource_id!r} in EXPRESSIONS'
        raise KeyError(msg) from None

    luts = ValidationLuts(
        disease=build_disease_lut(source['disease_path']),
        target=build_target_lut(source['target_path']),
        publication=build_publication_lut(source['publication_date_lut']),
    )

    postprocessor = EvidencePostprocessor(
        datasource_id=datasource_id,
        unique_fields=settings['unique_fields'],
        expressions=expressions,
        excluded_biotypes=settings.get('excluded_biotypes'),
    )

    processed = postprocessor.run(_read_evidence(source['evidence_path'], settings['evidence_format']), luts)

    # Two independent writes, not a persist-then-split like the pyspark implementation:
    # `.collect()`-ing once and splitting in memory doesn't bound memory, and the largest published
    # output (europepmc) is 9.03 GiB. This recomputes the upstream chain (LUT joins, hashing,
    # scoring, direction-of-effect) twice; an accepted trade for bounded memory, to revisit if
    # europepmc proves slow in practice.
    logger.info(f'writing valid evidence to {destination["evidence"]}')
    write_dataset(processed.valid, destination['evidence'])
    logger.info(f'writing failed evidence to {destination["failed_evidence"]}')
    write_dataset(processed.invalid, destination['failed_evidence'])
