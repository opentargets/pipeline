"""Map clinical report drug and disease labels to ids: ChEMBL curation, Open Targets dictionaries, drug NER.

Ported from mira's `map_entities` (`utils/mapping.py`), with OnToma replaced by the polars lookup tables of
`pts.transformers.ontoma_lut` and the polars `OnToma.map_entities` of `pts.transformers.utils.disease_mapping`.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.transformers.clinical.ner import extract_drug_entities
from pts.transformers.ontoma_lut.build import (
    DISEASE_COLUMNS,
    DRUG_COLUMNS,
    build_lut,
    disease_label_lut,
    drug_label_lut,
)
from pts.transformers.utils.disease_mapping.mapping import map_entities as map_lut_entities

#: The NER cache: each drug label NER has seen, with the drug names it extracted.
NER_CACHE_SCHEMA = pl.Schema({'query_label': pl.String, 'extracted_drugs': pl.List(pl.String)})


def _normalise_label_expr(col_name: str) -> pl.Expr:
    """Normalise entity labels for robust matching."""
    return pl.col(col_name).cast(pl.String).str.strip_chars().str.replace_all(r'\s+', ' ').str.to_lowercase()


def _apply_chembl_curation_mapping(
    df: pl.DataFrame,
    chembl_curation: pl.DataFrame,
    drug_column_name: str,
    disease_column_name: str,
    drug_id_column_name: str,
    disease_id_column_name: str,
) -> pl.DataFrame:
    """Map IDs directly from ChEMBL curation using study ID + label joins.

    For duplicate curated matches on the same key, keep all curated IDs and explode
    to one output row per curated ID combination.
    """
    required_df_cols = {'id', drug_column_name, disease_column_name}
    if not required_df_cols.issubset(df.columns):
        missing = sorted(required_df_cols.difference(df.columns))
        logger.warning(f'skipping chembl curation join, missing df columns: {missing}')
        return df

    required_curation_cols = {'studyId', 'drugFromSource', 'drugId', 'diseaseFromSource', 'diseaseId'}
    if not required_curation_cols.issubset(chembl_curation.columns):
        missing = sorted(required_curation_cols.difference(chembl_curation.columns))
        logger.warning(f'skipping chembl curation join, missing curation columns: {missing}')
        return df

    df_norm = df.with_columns(
        _normalise_label_expr('id').alias('__study_id_norm'),
        _normalise_label_expr(drug_column_name).alias('__drug_label_norm'),
        _normalise_label_expr(disease_column_name).alias('__disease_label_norm'),
    )

    curation_drug_lut = (
        chembl_curation
        .select('studyId', 'drugFromSource', 'drugId')
        .drop_nulls(['studyId', 'drugFromSource', 'drugId'])
        .with_columns(
            _normalise_label_expr('studyId').alias('__study_id_norm'),
            _normalise_label_expr('drugFromSource').alias('__drug_label_norm'),
            pl.col('drugId').alias('__curated_drug_id'),
        )
        .group_by(['__study_id_norm', '__drug_label_norm'])
        .agg(pl.col('__curated_drug_id').drop_nulls().unique().sort().alias('__curated_drug_ids'))
    )

    curation_disease_lut = (
        chembl_curation
        .select('studyId', 'diseaseFromSource', 'diseaseId')
        .drop_nulls(['studyId', 'diseaseFromSource', 'diseaseId'])
        .with_columns(
            _normalise_label_expr('studyId').alias('__study_id_norm'),
            _normalise_label_expr('diseaseFromSource').alias('__disease_label_norm'),
            pl.col('diseaseId').alias('__curated_disease_id'),
        )
        .group_by(['__study_id_norm', '__disease_label_norm'])
        .agg(pl.col('__curated_disease_id').drop_nulls().unique().sort().alias('__curated_disease_ids'))
    )

    joined = df_norm.join(
        curation_drug_lut,
        on=['__study_id_norm', '__drug_label_norm'],
        how='left',
        maintain_order='left',
    ).join(
        curation_disease_lut,
        on=['__study_id_norm', '__disease_label_norm'],
        how='left',
        maintain_order='left',
    )

    drug_filled_rows = joined.filter(
        pl.col(drug_id_column_name).is_null() & pl.col('__curated_drug_ids').is_not_null()
    ).height
    disease_filled_rows = joined.filter(
        pl.col(disease_id_column_name).is_null() & pl.col('__curated_disease_ids').is_not_null()
    ).height

    mapped = (
        joined
        .with_columns(
            pl
            .when(pl.col(drug_id_column_name).is_not_null())
            .then(pl.concat_list([pl.col(drug_id_column_name)]))
            .otherwise(pl.col('__curated_drug_ids'))
            .alias('__resolved_drug_ids'),
            pl
            .when(pl.col(disease_id_column_name).is_not_null())
            .then(pl.concat_list([pl.col(disease_id_column_name)]))
            .otherwise(pl.col('__curated_disease_ids'))
            .alias('__resolved_disease_ids'),
        )
        .with_columns(
            pl
            .when(pl.col('__resolved_drug_ids').is_null())
            .then(pl.lit([None], dtype=pl.List(pl.String)))
            .otherwise(pl.col('__resolved_drug_ids'))
            .alias('__resolved_drug_ids'),
            pl
            .when(pl.col('__resolved_disease_ids').is_null())
            .then(pl.lit([None], dtype=pl.List(pl.String)))
            .otherwise(pl.col('__resolved_disease_ids'))
            .alias('__resolved_disease_ids'),
        )
        .explode('__resolved_drug_ids', empty_as_null=True)
        .explode('__resolved_disease_ids', empty_as_null=True)
        .with_columns(
            pl.col('__resolved_drug_ids').alias(drug_id_column_name),
            pl.col('__resolved_disease_ids').alias(disease_id_column_name),
        )
        .drop(
            '__study_id_norm',
            '__drug_label_norm',
            '__disease_label_norm',
            '__curated_drug_ids',
            '__curated_disease_ids',
            '__resolved_drug_ids',
            '__resolved_disease_ids',
        )
    )

    logger.info(
        f'chembl curation direct join filled drugId for {drug_filled_rows} rows '
        f'and diseaseId for {disease_filled_rows} rows before explode'
    )

    return mapped


def _queries(df: pl.DataFrame, label_column: str, id_column: str, entity_type: str) -> pl.DataFrame:
    """The distinct labels in `label_column` of rows without an id, typed `entity_type`."""
    return (
        df
        .filter(pl.col(label_column).is_not_null() & pl.col(id_column).is_null())
        .select(pl.col(label_column).alias('query_label'), pl.lit(entity_type).alias('entity_type'))
        .unique()
    )


def _read_ner_cache(path: str | None) -> pl.DataFrame:
    """The NER cache at `path`; empty when there is no path or it cannot be read."""
    if path is not None:
        try:
            cache = pl.read_parquet(path).select(NER_CACHE_SCHEMA.names()).cast(NER_CACHE_SCHEMA)
        except Exception:
            logger.info(f'no readable ner cache at {path}, will compute all labels')
        else:
            logger.info(f'loaded ner cache from {path}')
            return cache
    return pl.DataFrame(schema=NER_CACHE_SCHEMA)


def _ner_drug_ids(labels: pl.Series, drug_lut: pl.LazyFrame, cache_path: str | None, batch_size: int) -> pl.DataFrame:
    """Drug ids of the names NER extracts from each label, from the cache where it has the label.

    Labels missing from the cache are extracted and added to it.
    """
    cached = _read_ner_cache(cache_path)
    new_labels = labels.filter(~labels.is_in(cached['query_label'].implode()))
    if new_labels.len() > 0:
        logger.info(
            f'applying ner to {new_labels.len()} new drug labels (using cache for {labels.len() - new_labels.len()})'
        )
        new = pl.DataFrame(
            {'query_label': new_labels, 'extracted_drugs': extract_drug_entities(new_labels, batch_size)},
            schema=NER_CACHE_SCHEMA,
        )
        cached = pl.concat([cached, new])
        if cache_path is not None:
            logger.info(f'updating ner cache: {cache_path}')
            cached.write_parquet(cache_path, mkdir=True)
    else:
        logger.info(f'all {labels.len()} labels found in ner cache, skipping ner extraction')

    extracted = (
        cached
        .filter(pl.col('query_label').is_in(labels.implode()))
        .select('query_label', pl.col('extracted_drugs').alias('clean_label'))
        .explode('clean_label')
        .filter(pl.col('clean_label').is_not_null())
    )
    mapped = map_lut_entities(
        extracted.lazy(), drug_lut, 'mapped_ids', 'clean_label', 'label', type_col=pl.lit('CD')
    ).collect()
    return (
        mapped
        .filter(pl.col('mapped_ids').is_not_null())
        .group_by('query_label')
        .agg(pl.col('mapped_ids').list.explode(keep_nulls=False, empty_as_null=False).unique().sort())
        .rename({'mapped_ids': 'ner_mapped_ids'})
    )


def map_entities(
    df: pl.DataFrame,
    disease: pl.DataFrame,
    drug: pl.DataFrame,
    chembl_curation: pl.DataFrame | None,
    ner_cache_path: str | None,
    ner_batch_size: int,
    drug_column_name: str = 'drugFromSource',
    disease_column_name: str = 'diseaseFromSource',
    drug_id_column_name: str = 'drugId',
    disease_id_column_name: str = 'diseaseId',
) -> pl.DataFrame:
    """Map drug and disease labels to ids, keeping ids already set.

    In order of precedence:
    1. ids already in `df`, then ids curated by ChEMBL for the study and label;
    2. the disease and drug label dictionaries, for labels still without an id;
    3. drug NER, for drug labels still without an id: the extracted names mapped with the drug dictionary.

    Args:
        df: Reports to map.
        disease: Disease index, with the `DISEASE_COLUMNS` of `output/disease`.
        drug: Drug molecules, with the `DRUG_COLUMNS` of `intermediate/chembl_molecule`.
        chembl_curation: ChEMBL's clinical trial curation, by `studyId` and source labels.
        ner_cache_path: Parquet file of NER results, read and rewritten with the labels it lacked.
        ner_batch_size: Batch size for drug NER.
        drug_column_name: Column of drug labels.
        disease_column_name: Column of disease labels.
        drug_id_column_name: Column of drug ids, added when missing.
        disease_id_column_name: Column of disease ids, added when missing.

    Returns:
        `df` with the ids found, one row per drug id and disease id of a row.
    """
    missing_id_cols = [
        pl.lit(None, dtype=pl.String).alias(column)
        for column in (disease_id_column_name, drug_id_column_name)
        if column not in df.columns
    ]
    if missing_id_cols:
        df = df.with_columns(missing_id_cols)

    if chembl_curation is not None:
        df = _apply_chembl_curation_mapping(
            df=df,
            chembl_curation=chembl_curation,
            drug_column_name=drug_column_name,
            disease_column_name=disease_column_name,
            drug_id_column_name=drug_id_column_name,
            disease_id_column_name=disease_id_column_name,
        )

    queries = pl.concat([
        _queries(df, disease_column_name, disease_id_column_name, 'DS'),
        _queries(df, drug_column_name, drug_id_column_name, 'CD'),
    ])
    if queries.is_empty():
        logger.info('all entities already mapped after curation and existing IDs')
        return df

    lut = build_lut([
        disease_label_lut(disease.select(DISEASE_COLUMNS)),
        drug_label_lut(drug.select(DRUG_COLUMNS)),
    ]).lazy()
    mapping_results = map_lut_entities(
        queries.lazy(), lut, 'mapped_ids', 'query_label', 'label', type_col_name='entity_type'
    ).collect()

    unmapped_drugs = (
        mapping_results
        .filter((pl.col('entity_type') == 'CD') & pl.col('mapped_ids').is_null())
        .get_column('query_label')
        .unique()
        .sort()
    )
    if unmapped_drugs.len() > 0:
        logger.info(f'found {unmapped_drugs.len()} unmapped drug labels')
        ner = _ner_drug_ids(unmapped_drugs, lut.filter(pl.col('entityType') == 'CD'), ner_cache_path, ner_batch_size)
        mapping_results = (
            mapping_results
            .join(ner, on='query_label', how='left', maintain_order='left')
            .with_columns(
                pl
                .when(pl.col('mapped_ids').is_null() & (pl.col('entity_type') == 'CD'))
                .then(pl.col('ner_mapped_ids'))
                .otherwise(pl.col('mapped_ids'))
                .alias('mapped_ids')
            )
            .drop('ner_mapped_ids')
        )
        logger.info(f'ner recovered {ner.height}/{unmapped_drugs.len()}')

    mapping_results = mapping_results.filter(pl.col('mapped_ids').is_not_null()).select(
        'query_label',
        'entity_type',
        pl.when(pl.col('entity_type') == 'DS').then(pl.col('mapped_ids')).alias('diseaseIds'),
        pl.when(pl.col('entity_type') == 'CD').then(pl.col('mapped_ids')).alias('drugIds'),
    )

    return (
        df
        .with_columns(
            pl
            .when(pl.col(disease_id_column_name).is_null())
            .then(pl.col(disease_column_name))
            .alias('__disease_query'),
            pl.when(pl.col(drug_id_column_name).is_null()).then(pl.col(drug_column_name)).alias('__drug_query'),
        )
        .join(
            mapping_results.filter(pl.col('entity_type') == 'DS').select(['query_label', 'diseaseIds']),
            left_on='__disease_query',
            right_on='query_label',
            how='left',
            suffix='_disease',
            maintain_order='left',
        )
        .join(
            mapping_results.filter(pl.col('entity_type') == 'CD').select(['query_label', 'drugIds']),
            left_on='__drug_query',
            right_on='query_label',
            how='left',
            suffix='_drug',
            maintain_order='left',
        )
        .drop('__disease_query', '__drug_query')
        .explode('diseaseIds', empty_as_null=True)
        .explode('drugIds', empty_as_null=True)
        .rename({'diseaseIds': f'new_{disease_id_column_name}', 'drugIds': f'new_{drug_id_column_name}'})
        .with_columns(
            pl.coalesce(pl.col(disease_id_column_name), pl.col(f'new_{disease_id_column_name}')).alias(
                disease_id_column_name
            ),
            pl.coalesce(pl.col(drug_id_column_name), pl.col(f'new_{drug_id_column_name}')).alias(drug_id_column_name),
        )
        .drop(f'new_{disease_id_column_name}', f'new_{drug_id_column_name}')
    )
