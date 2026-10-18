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
IRRELEVANT_CROSSREF_PREFIXES = (
    'PMID',
    'DOI:',
    'ORCID',
    'PERSON',
    'ISBN',
    'WIKIPEDIA',
    'HTTP',
    'QUANT',
    'UM-BBD_PATHWAYID',
)
#: (column, entityScore, entitySource, nlpPipelineTrack) of each target label `OpenTargetsTarget.as_label_lut` reads.
TARGET_LABELS = (
    ('approvedName', 1.0, 'approved_name', 'term'),
    ('approvedSymbol', 1.0, 'approved_symbol', 'symbol'),
    ('nameSynonyms', 0.999, 'name_synonym', 'term'),
    ('symbolSynonyms', 0.999, 'symbol_synonym', 'symbol'),
    ('proteinIds', 0.999, 'protein_id', 'symbol'),
    ('obsoleteNames', 0.998, 'obsolete_name', 'term'),
    ('obsoleteSymbols', 0.998, 'obsolete_symbol', 'symbol'),
)
#: (column, entityScore, entitySource) of each drug label `OpenTargetsDrug.as_label_lut` reads, all on the symbol track.
DRUG_LABELS = (
    ('name', 1.0, 'name'),
    ('tradeNames', 0.999, 'trade_name'),
    ('synonymsCurated', 0.999, 'synonym'),
    ('synonymsInferred', 0.998, 'synonym_aact'),
    ('combinationProducts', 0.999, 'trade_name_component'),
)
#: Drug cross-reference sources whose ids are labels.
DRUG_LABEL_CROSSREF_SOURCES = ('DailyMed', 'USAN', 'EMA')

_CURATION_ID = rf'^http{JAVA_DOT}+/([0-9A-Za-z_]+_[0-9A-Za-z_]+){JAVA_END}'
_NUMBERED_ITEM = rf'^[12]\){JAVA_END}'
_JAVA_SPACE = r' \t\n\x0B\f\r'
_COMPONENT_OF = '(?i) component of '
_COMBINATION_PRODUCT = rf'(?i)^{JAVA_DOT}+ component of ({JAVA_DOT}+){JAVA_END}'
_EMA_LABEL = rf'{JAVA_DOT}+/EPAR/({JAVA_DOT}+)'


def _entities(frame: pl.DataFrame, columns: tuple[tuple[str, float, str], ...]) -> pl.DataFrame:
    """`annotate_entity` and explode: one row per non-null label of each column, with its score and source."""
    frames = []
    for column, score, source in columns:
        entities = frame.select(
            pl.col('id').alias('entityId'),
            pl.col(column).alias('entityLabel'),
            pl.lit(score, dtype=pl.Float64).alias('entityScore'),
            pl.lit(source).alias('entitySource'),
        )
        if isinstance(frame.schema[column], pl.List):
            entities = entities.explode('entityLabel')
        frames.append(entities)
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


def _label_lut(entities: pl.DataFrame, entity_type: str) -> pl.DataFrame:
    """Trimmed labels in both translations, as a `RAW_ENTITY_LUT` of `entity_type` labels."""
    return _raw_lut(
        entities.with_columns(
            entityLabel=alternative_translations(spark_trim(pl.col('entityLabel'))),
            entityType=pl.lit(entity_type),
            entityKind=pl.lit('label'),
        ).explode('entityLabel')
    )


def disease_label_lut(disease: pl.DataFrame) -> pl.DataFrame:
    """`OpenTargetsDisease.as_label_lut`: disease names and synonyms, each in both translations."""
    entities = _entities(disease, DISEASE_LABELS)
    return _label_lut(entities.with_columns(nlpPipelineTrack=determine_track(pl.col('entityLabel'))), 'DS')


def disease_id_lut(disease: pl.DataFrame) -> pl.DataFrame:
    """`OpenTargetsDisease.as_id_lut`: disease ids, obsolete ids and cross-references, formatted."""
    label = spark_upper(spark_trim(pl.col('entityLabel')))
    return _raw_lut(
        _entities(disease, DISEASE_IDS)
        .with_columns(entityLabel=label)
        .filter(~pl.col('entityLabel').str.contains_any(list(IRRELEVANT_CROSSREF_PREFIXES)))
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
        .agg(pl.col('entityId').drop_nulls().unique())
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


def target_label_lut(target: pl.DataFrame) -> pl.DataFrame:
    """`OpenTargetsTarget.as_label_lut`: target names, symbols, synonyms and protein ids, each in both translations."""
    labels = target.with_columns(
        pl.col(column).list.eval(pl.element().struct.field('id' if column == 'proteinIds' else 'label'))
        for column, *_ in TARGET_LABELS
        if isinstance(target.schema[column], pl.List)
    )
    entities = pl.concat(
        _entities(labels, ((column, score, source),)).with_columns(nlpPipelineTrack=pl.lit(track))
        for column, score, source, track in TARGET_LABELS
    )
    return _label_lut(entities, 'GP')


def _synonyms(source: str) -> pl.Expr:
    """The labels of the drug synonyms from `source`."""
    synonym = pl.element().struct
    return pl.col('synonyms').list.eval(synonym.field('label').filter(synonym.field('source') == source))


def _combination_product(label: pl.Expr) -> pl.Expr:
    """`extract_combination_product`: `Y` of an `X component of Y` label, cleaned, else empty."""
    product = regexp_extract(label, _COMBINATION_PRODUCT, 1)
    product = product.str.replace(rf'^[{_JAVA_SPACE}/,;:-]+', '').str.replace(rf'[{_JAVA_SPACE}/,;:-]+{JAVA_END}', '')
    return spark_trim(product.str.replace_all(rf'[{_JAVA_SPACE}]+', ' '))


def _drug_crossref_labels(drug: pl.DataFrame) -> pl.DataFrame:
    """DailyMed and USAN ids with `%20` as a space, and EMA ids after `/EPAR/`; none for a drug with a null id list."""
    ids = pl.col('ids')
    return (
        drug
        .select('id', 'crossReferences')
        .with_row_index('drug')
        .explode('crossReferences')
        .unnest('crossReferences')
        .filter(pl.col('source').is_in(DRUG_LABEL_CROSSREF_SOURCES))
        .filter(ids.is_not_null().all().over('drug'))
        .explode('ids')
        .select(
            pl.col('id').alias('entityId'),
            pl
            .when(pl.col('source') == 'EMA')
            .then(regexp_extract(ids, _EMA_LABEL, 1))
            .otherwise(ids.str.replace_all('%20', ' ', literal=True))
            .alias('entityLabel'),
            pl.lit(0.997, dtype=pl.Float64).alias('entityScore'),
            pl.lit('crossref').alias('entitySource'),
        )
    )


def drug_label_lut(drug: pl.DataFrame) -> pl.DataFrame:
    """`OpenTargetsDrug.as_label_lut`: drug names, trade names, synonyms, combination products and label crossrefs."""
    lists = ('tradeNames', 'synonymsCurated', 'synonymsInferred')
    labelled = drug.select(
        'id',
        'name',
        'crossReferences',
        pl.col('tradeNames').list.eval(pl.element().struct.field('label')),
        _synonyms('ChEMBL').alias('synonymsCurated'),
        _synonyms('AACT').alias('synonymsInferred'),
    ).filter(
        ~pl.col('name').str.to_lowercase().str.starts_with('chembl')
        | (pl.col('tradeNames').list.len() > 0)
        | (pl.col('synonymsCurated').list.len() > 0)
        | (pl.col('synonymsInferred').list.len() > 0)
    )
    empty = pl.lit([], dtype=pl.List(pl.String))
    labelled = labelled.with_columns(
        combinationProducts=pl.concat_list(pl.col(column).fill_null(empty) for column in lists).list.eval(
            _combination_product(pl.element())
        ),
        **{
            column: pl.col(column).list.eval(pl.element().filter(~pl.element().str.contains(_COMPONENT_OF)))
            for column in lists
        },
    )
    entities = pl.concat([_entities(labelled, DRUG_LABELS), _drug_crossref_labels(labelled)])
    return _label_lut(entities.with_columns(nlpPipelineTrack=pl.lit('symbol')), 'CD')
