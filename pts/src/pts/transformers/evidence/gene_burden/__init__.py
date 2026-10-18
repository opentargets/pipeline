"""Gene burden evidence.

Parses every sub-source into one union, maps diseases with the OnToma lookup tables through
`add_efo_mapping`, by label and by id, falling back to the id the source gives, and post-processes it.
"""

from __future__ import annotations

from typing import Any

import polars as pl
from loguru import logger
from otter.config.model import Config

from pts.schemas.gene_burden import GENE_BURDEN_UNION
from pts.transformers.evidence.gene_burden.az import parse_az
from pts.transformers.evidence.gene_burden.brava import parse_brava
from pts.transformers.evidence.gene_burden.curated import parse_curated
from pts.transformers.evidence.gene_burden.finngen import parse_finngen
from pts.transformers.evidence.gene_burden.genebass import parse_genebass
from pts.transformers.evidence.gene_burden.genes_and_health import parse_genes_and_health
from pts.transformers.evidence.utils.datasource import EvidenceDatasource
from pts.transformers.evidence.utils.expressions import log10_rescaled_score
from pts.transformers.evidence.utils.luts import build_luts
from pts.transformers.evidence.utils.ontology import MAPPED_ID_COLUMN, add_efo_mapping
from pts.transformers.evidence.utils.run import write_evidence
from pts.transformers.utils.spark_frames import spark_distinct
from pts.transformers.utils.spark_math import normalise_mantissa

CURATED_ID_COLUMN = 'curatedDiseaseFromSourceMappedId'

GENE_BURDEN = EvidenceDatasource(
    datasource_id='gene_burden',
    # order feeds the id, as in the spark config
    unique_fields=[
        'targetId',
        'targetFromSourceId',
        'diseaseId',
        'datasourceId',
        'ancestryId',
        'diseaseFromSource',
        'statisticalMethod',
        'projectId',
    ],
    score=log10_rescaled_score(pl.col('resourceScore'), 0.25, 1e-7, 1e-17),
)
# a NaN compares above every number, as in spark
DIRECTION_ON_TRAIT = (
    pl
    .when(pl.col('oddsRatio') > 1.0)
    .then(pl.lit('risk'))
    .when(pl.col('oddsRatio') < 1.0)
    .then(pl.lit('protect'))
    .when(pl.col('beta') > 0.0)
    .then(pl.lit('risk'))
    .when(pl.col('beta') < 0.0)
    .then(pl.lit('protect'))
    .otherwise(None)
)
DIRECTION_ON_TARGET = pl.lit('LoF')


def parse_sub_sources(source: dict[str, str], settings: dict[str, Any]) -> list[pl.LazyFrame]:
    """Every sub-source's evidence in `GENE_BURDEN_PARSED`, in the order of the union."""
    return [
        parse_az(source['az_binary'], source['az_quantitative'], source['az_genes'], source['az_phenotypes']),
        parse_curated(source['curated_studies']),
        parse_genebass(source['genebass']),
        parse_finngen(source['finngen'], source['finngen_phenotypes'], settings['finngen_release']),
        parse_genes_and_health(source['genes_and_health']),
        parse_brava(source['brava']),
    ]


def _project_counts(frame: pl.DataFrame) -> dict[str | None, int]:
    return dict(frame.group_by('projectId').len().iter_rows())


def _format_counts(counts: dict[str | None, int]) -> str:
    projects = sorted((p for p, n in counts.items() if n), key=lambda p: (p is None, p or ''))
    return ', '.join(f'{project}: {counts[project]}' for project in projects)


def assemble(parsed: list[pl.LazyFrame]) -> pl.DataFrame:
    """The distinct union of the sub-sources, every mantissa of 10 or more normalised.

    The normalised rows, and those of them merged with an identical row by the distinct, are
    counted by `projectId` and logged.
    """
    union = pl.concat(parsed, how='diagonal_relaxed').collect()
    shifted = union.filter((pl.col('pValueMantissa') >= 10) & ~pl.col('pValueMantissa').is_nan())
    logger.info(
        f'{shifted.height} p-value mantissas of 10 or more normalised, by project: '
        f'{_format_counts(_project_counts(shifted))}'
    )
    mantissa, exponent = normalise_mantissa(pl.col('pValueMantissa'), pl.col('pValueExponent'))
    distinct = spark_distinct(union.with_columns(mantissa.alias('pValueMantissa'), exponent.alias('pValueExponent')))
    before, after = _project_counts(spark_distinct(union)), _project_counts(distinct)
    merged = {project: n - after.get(project, 0) for project, n in before.items()}
    logger.info(
        f'{sum(merged.values())} normalised rows merged with an identical row, by project: {_format_counts(merged)}'
    )
    return distinct


def check_union(frame: pl.DataFrame) -> pl.DataFrame:
    """`frame`, checked to be in `GENE_BURDEN_UNION`.

    Raises:
        ValueError: if it is not.
    """
    if frame.schema != GENE_BURDEN_UNION:
        msg = f'gene burden evidence is not in GENE_BURDEN_UNION: {frame.schema}'
        raise ValueError(msg)
    return frame


def map_diseases(union: pl.DataFrame, disease_label_lut: str, disease_id_lut: str) -> pl.DataFrame:
    """Map diseases by label and source id; the source's own disease id is used only where neither maps.

    Rows where the mapping overrides a different id given by the source are counted by `projectId`
    and logged.
    """
    mapped = add_efo_mapping(
        union.lazy(), disease_label_lut_path=disease_label_lut, disease_id_lut_path=disease_id_lut
    ).collect()
    mapped_id, curated_id = pl.col(MAPPED_ID_COLUMN), pl.col(CURATED_ID_COLUMN)
    overridden = mapped_id.is_not_null() & curated_id.is_not_null() & (mapped_id != curated_id)
    logger.info(
        f'{mapped.filter(overridden).height} source disease ids overridden by the mapping, by project: '
        f'{_format_counts(_project_counts(mapped.filter(overridden)))}'
    )
    coalesced = mapped.with_columns(pl.coalesce(mapped_id, curated_id).alias(MAPPED_ID_COLUMN))
    return check_union(coalesced.drop(CURATED_ID_COLUMN))


def gene_burden(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    config: Config,
) -> None:
    """Generate and post-process gene burden evidence.

    Args:
        source: the sub-source inputs (`az_binary`, `az_quantitative`, `az_genes`, `az_phenotypes`,
            `curated_studies`, `genebass`, `finngen`, `finngen_phenotypes`, `genes_and_health`,
            `brava`), `ontoma_disease_label_lut`, `ontoma_disease_id_lut`, `disease`, `target` and
            `literature` locations.
        destination: `evidence` and `failed_evidence` output directories.
        settings: `finngen_release`, the FinnGen `releaseVersion`.
        config: unused.
    """
    logger.info(f'load gene burden sub-sources from {source}')
    union = assemble(parse_sub_sources(source, settings))
    raw = map_diseases(union, source['ontoma_disease_label_lut'], source['ontoma_disease_id_lut'])
    luts = build_luts(raw.lazy(), disease=source['disease'], target=source['target'], literature=source['literature'])
    evidence = (
        GENE_BURDEN
        .postprocess(raw.lazy(), luts)
        .assign_direction_on_trait(DIRECTION_ON_TRAIT)
        .assign_direction_on_target(DIRECTION_ON_TARGET, luts.target)
        .hash_long_variant_identifiers()
    )
    write_evidence(evidence, destination)
