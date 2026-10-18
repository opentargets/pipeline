"""Per-source raw lookup tables: polars ports of OnToma's `as_label_lut` and `as_id_lut` builders."""

from __future__ import annotations

import polars as pl

from pts.schemas.ontoma_lut import RAW_ENTITY_LUT
from pts.transformers.utils.disease_mapping.query import (
    JAVA_DOT,
    JAVA_END,
    alternative_translations,
    clean_disease_label,
    determine_track,
    format_identifier,
)
from pts.transformers.utils.spark_strings import regexp_extract, spark_trim, spark_upper

#: (column, entityScore, entitySource) of each disease label `OpenTargetsDisease.as_label_lut` reads.
DISEASE_LABELS = (
    ('name', 1.0, 'name'),
    ('exactSynonyms', 0.999, 'exact_synonym'),
    ('narrowSynonyms', 0.998, 'narrow_synonym'),
    ('broadSynonyms', 0.997, 'broad_synonym'),
    ('relatedSynonyms', 0.996, 'related_synonym'),
)
#: (column, entityScore, entitySource) of each disease id `OpenTargetsDisease.as_id_lut` reads.
DISEASE_IDS = (
    ('id', 1.0, 'id'),
    ('obsoleteTerms', 0.999, 'obsolete_id'),
    ('dbXRefs', 0.998, 'crossref'),
    ('obsoleteXRefs', 0.997, 'obsolete_crossref'),
)
#: `filter_disease_crossrefs`: an uppercased id containing any of these is dropped.
IRRELEVANT_CROSSREF_PREFIXES = [
    'PMID',
    'DOI:',
    'ORCID',
    'PERSON',
    'ISBN',
    'WIKIPEDIA',
    'HTTP',
    'QUANT',
    'UM-BBD_PATHWAYID',
]

_CURATION_ID = rf'^http{JAVA_DOT}+/([0-9A-Za-z_]+_[0-9A-Za-z_]+){JAVA_END}'
_NUMBERED_ITEM = rf'^[12]\){JAVA_END}'


def _entities(disease: pl.DataFrame, columns: tuple[tuple[str, float, str], ...]) -> pl.DataFrame:
    """`annotate_entity` and explode: one row per non-null label of each column, with its score and source."""
    frames = []
    for column, score, source in columns:
        frame = disease.select(
            pl.col('id').alias('entityId'),
            pl.col(column).alias('entityLabel'),
            pl.lit(score, dtype=pl.Float64).alias('entityScore'),
            pl.lit(source).alias('entitySource'),
        )
        if isinstance(disease.schema[column], pl.List):
            frame = frame.explode('entityLabel')
        frames.append(frame)
    return pl.concat(frames).filter(pl.col('entityLabel').is_not_null())


def _raw_lut(df: pl.DataFrame) -> pl.DataFrame:
    """Rows with a non-empty label, distinct and sorted, as `RAW_ENTITY_LUT`."""
    return (
        df
        .filter(pl.col('entityLabel').str.len_chars() > 0)
        .select(pl.col(name).cast(dtype) for name, dtype in RAW_ENTITY_LUT.items())
        .unique()
        .sort(RAW_ENTITY_LUT.names())
    )


def disease_label_lut(disease: pl.DataFrame) -> pl.DataFrame:
    """`OpenTargetsDisease.as_label_lut`: disease names and synonyms, each in both translations."""
    return _raw_lut(
        _entities(disease, DISEASE_LABELS)
        .with_columns(
            nlpPipelineTrack=determine_track(pl.col('entityLabel')),
            entityLabel=alternative_translations(spark_trim(pl.col('entityLabel'))),
            entityType=pl.lit('DS'),
            entityKind=pl.lit('label'),
        )
        .explode('entityLabel')
    )


def disease_id_lut(disease: pl.DataFrame) -> pl.DataFrame:
    """`OpenTargetsDisease.as_id_lut`: disease ids, obsolete ids and cross-references, formatted."""
    label = spark_upper(spark_trim(pl.col('entityLabel')))
    return _raw_lut(
        _entities(disease, DISEASE_IDS)
        .with_columns(entityLabel=label)
        .filter(~pl.col('entityLabel').str.contains_any(IRRELEVANT_CROSSREF_PREFIXES))
        .with_columns(
            entityLabel=format_identifier(pl.col('entityLabel')),
            nlpPipelineTrack=pl.lit('symbol'),
            entityType=pl.lit('DS'),
            entityKind=pl.lit('id'),
        )
    )


def _remapped_ids(disease: pl.DataFrame) -> pl.DataFrame:
    """Each disease-index id, obsolete id or cross-reference whose top-scored rows name exactly one disease."""
    return (
        disease_id_lut(disease)
        .filter(pl.col('entityScore') == pl.col('entityScore').max().over('entityLabel'))
        .group_by('entityLabel')
        .agg(pl.col('entityId').unique())
        .filter(pl.col('entityId').list.len() == 1)
        .select(
            pl.col('entityId').list.first(),
            pl.col('entityLabel').str.to_lowercase().str.replace_all(':', '_', literal=True).alias('entityIdToJoin'),
        )
    )


def curation_label_lut(curation: pl.DataFrame, source: str, disease: pl.DataFrame) -> pl.DataFrame:
    """`DiseaseCuration.as_label_lut`, remapping obsolete mappings: curated labels of ids resolving to one disease."""
    label = clean_disease_label(spark_trim(pl.col('PROPERTY_VALUE')))
    curated = curation.select(
        regexp_extract(pl.col('SEMANTIC_TAG'), _CURATION_ID, 1).str.to_lowercase().alias('entityIdToJoin'),
        alternative_translations(label).alias('entityLabel'),
        pl.lit(1.0, dtype=pl.Float64).alias('entityScore'),
        determine_track(label).alias('nlpPipelineTrack'),
        pl.lit(source).alias('entitySource'),
        pl.lit('DS').alias('entityType'),
        pl.lit('label').alias('entityKind'),
    ).explode('entityLabel')
    return _raw_lut(
        curated
        .join(_remapped_ids(disease), on='entityIdToJoin', how='inner')
        .filter(~pl.col('entityLabel').str.contains(_NUMBERED_ITEM))
        .filter(pl.col('entityId').str.len_chars() > 0)
    )
