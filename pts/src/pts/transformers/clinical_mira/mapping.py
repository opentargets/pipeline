"""Map clinical report drug and disease labels to ids: ChEMBL curation, Open Targets dictionaries, staged drug NER.

Ported from mira's `map_entities` (`utils/mapping.py`), with OnToma replaced by the polars lookup tables of
`pts.transformers.ontoma_lut` and the polars `OnToma.map_entities` of `pts.transformers.utils.disease_mapping`.
"""

from __future__ import annotations

import polars as pl
from loguru import logger

from pts.transformers.ontoma_lut.build import (
    DISEASE_COLUMNS,
    DRUG_COLUMNS,
    build_lut,
    disease_label_lut,
    drug_label_lut,
)
from pts.transformers.utils.disease_mapping.mapping import map_entities as map_lut_entities


def require_staged(texts: pl.Series, staged: pl.Series, what: str) -> None:
    """Check that every non-null text is staged.

    Raises:
        ValueError: naming `what`, how many distinct texts are not staged and up to 5 of them.
    """
    distinct = texts.drop_nulls().unique()
    missing = distinct.filter(~distinct.is_in(staged.implode())).sort()
    if missing.len() > 0:
        raise ValueError(f'{missing.len()} {what} not staged, e.g. {missing.head(5).to_list()}')


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


def _staged_ner_names(labels: pl.Series, staged_drug_entities: pl.DataFrame) -> pl.DataFrame:
    """The names staged for each label, as `query_label` and `extracted_drugs`.

    Raises:
        ValueError: if a label is not staged.
    """
    require_staged(labels, staged_drug_entities['label'], 'drug labels')
    return staged_drug_entities.select(pl.col('label').alias('query_label'), 'extracted_drugs').filter(
        pl.col('query_label').is_in(labels.implode())
    )


def _ner_drug_ids(names: pl.DataFrame, drug_lut: pl.LazyFrame) -> pl.DataFrame:
    """Drug ids of the extracted names of each label, mapped with the drug dictionary."""
    extracted = (
        names
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
    staged_drug_entities: pl.DataFrame,
    drug_column_name: str = 'drugFromSource',
    disease_column_name: str = 'diseaseFromSource',
    drug_id_column_name: str = 'drugId',
    disease_id_column_name: str = 'diseaseId',
    record_column_name: str = 'id',
) -> pl.DataFrame:
    """Map drug and disease labels to ids, keeping ids already set.

    In order of precedence:
    1. ids already in `df`, then ids curated by ChEMBL for the study and label;
    2. the disease and drug label dictionaries, for labels still without an id;
    3. drug NER, for drug labels still without an id: the staged names of the label mapped with the drug dictionary.

    A disease id the dictionary flags as acronym-only is kept only when another row of the same record has it
    without the flag, from the dictionary, ChEMBL curation or `df`.

    Args:
        df: Reports to map.
        disease: Disease index, with the `DISEASE_COLUMNS` of `output/disease`.
        drug: Drug molecules, with the `DRUG_COLUMNS` of `intermediate/chembl_molecule`.
        chembl_curation: ChEMBL's clinical trial curation, by `studyId` and source labels.
        staged_drug_entities: Drug names staged per drug label, as `label` and `extracted_drugs`.
        drug_column_name: Column of drug labels.
        disease_column_name: Column of disease labels.
        drug_id_column_name: Column of drug ids, added when missing.
        disease_id_column_name: Column of disease ids, added when missing.
        record_column_name: Column identifying the report whose rows give an acronym its context.

    Returns:
        `df` with the ids found, one row per drug id and disease id of a row.

    Raises:
        ValueError: if a drug label reaching NER is not staged.
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
    disease_results = (
        map_lut_entities(
            queries.filter(pl.col('entity_type') == 'DS').lazy(),
            lut,
            'mapped_ids',
            'query_label',
            'label',
            type_col_name='entity_type',
            include_entity_source=True,
        )
        .collect()
        .pipe(_disease_ids_with_acronym_flag)
    )
    drug_results = map_lut_entities(
        queries.filter(pl.col('entity_type') == 'CD').lazy(),
        lut,
        'mapped_ids',
        'query_label',
        'label',
        type_col_name='entity_type',
    ).collect()

    unmapped_drugs = drug_results.filter(pl.col('mapped_ids').is_null()).get_column('query_label').unique().sort()
    if unmapped_drugs.len() > 0:
        logger.info(f'found {unmapped_drugs.len()} unmapped drug labels')
        names = _staged_ner_names(unmapped_drugs, staged_drug_entities)
        ner = _ner_drug_ids(names, lut.filter(pl.col('entityType') == 'CD'))
        drug_results = (
            drug_results
            .join(ner, on='query_label', how='left', maintain_order='left')
            .with_columns(pl.coalesce('mapped_ids', 'ner_mapped_ids').alias('mapped_ids'))
            .drop('ner_mapped_ids')
        )
        logger.info(f'ner recovered {ner.height}/{unmapped_drugs.len()}')

    columns = df.columns
    new_disease = f'new_{disease_id_column_name}'
    new_drug = f'new_{drug_id_column_name}'
    mapped = (
        df
        .with_row_index('__row')
        .with_columns(
            pl
            .when(pl.col(disease_id_column_name).is_null())
            .then(pl.col(disease_column_name))
            .alias('__disease_query'),
            pl.when(pl.col(drug_id_column_name).is_null()).then(pl.col(drug_column_name)).alias('__drug_query'),
        )
        .join(
            disease_results.filter(pl.col('mapped_ids').is_not_null()).select(
                pl.col('query_label').alias('__disease_query'), pl.col('mapped_ids').alias(new_disease)
            ),
            on='__disease_query',
            how='left',
            maintain_order='left',
        )
        .join(
            drug_results.filter(pl.col('mapped_ids').is_not_null()).select(
                pl.col('query_label').alias('__drug_query'), pl.col('mapped_ids').alias(new_drug)
            ),
            on='__drug_query',
            how='left',
            maintain_order='left',
        )
        .explode(new_disease, empty_as_null=True)
        .explode(new_drug, empty_as_null=True)
        .rechunk()
        .with_columns(
            pl.col(new_disease).struct.field('entityId').alias('__lut_disease_id'),
            pl.col(new_disease).struct.field('isAcronym').fill_null(False).alias('__acronym'),
        )
        .with_columns(
            pl.coalesce(disease_id_column_name, '__lut_disease_id').alias(disease_id_column_name),
            pl.coalesce(drug_id_column_name, new_drug).alias(drug_id_column_name),
        )
    )
    return _drop_acronyms_without_context(
        mapped, disease_id_column_name, drug_id_column_name, record_column_name
    ).select(columns)


def _disease_ids_with_acronym_flag(results: pl.DataFrame) -> pl.DataFrame:
    """`mapped_ids` as sorted `{entityId, isAcronym}` structs, an id acronym-only if all of its structs are."""
    ids = (
        results
        .select('query_label', pl.col('mapped_ids').alias('id'))
        .explode('id')
        .filter(pl.col('id').is_not_null())
        .unnest('id')
        .group_by('query_label', 'entityId')
        .agg(pl.col('isAcronym').fill_null(False).all())
        .sort('query_label', 'entityId')
        .group_by('query_label', maintain_order=True)
        .agg(pl.struct('entityId', 'isAcronym').alias('mapped_ids'))
    )
    return results.select('query_label').join(ids, on='query_label', how='left', maintain_order='left')


def _drop_acronyms_without_context(
    mapped: pl.DataFrame, disease_id_column_name: str, drug_id_column_name: str, record_column_name: str
) -> pl.DataFrame:
    """Drop acronym-only disease ids that no non-acronym row of the same record names.

    An input row (`__row`) left without a disease id keeps one row per drug id, with a null disease id.
    """
    disease_id = pl.col(disease_id_column_name)
    context = (
        mapped
        .filter(~pl.col('__acronym') & disease_id.is_not_null())
        .select(record_column_name, disease_id_column_name, pl.lit(True).alias('__in_context'))
        .unique()
    )
    flagged = mapped.join(
        context, on=[record_column_name, disease_id_column_name], how='left', nulls_equal=True, maintain_order='left'
    ).with_columns((pl.col('__acronym') & pl.col('__in_context').is_null()).alias('__drop'))
    dropped = flagged.filter('__drop').select('__row', disease_id_column_name).unique().height
    logger.info(f'dropped {dropped} acronym disease ids without report context')
    all_dropped = pl.col('__drop').all().over('__row')
    return (
        flagged
        .filter(~pl.col('__drop') | all_dropped)
        .with_columns(pl.when(all_dropped).then(None).otherwise(disease_id).alias(disease_id_column_name))
        .unique(['__row', disease_id_column_name, drug_id_column_name], maintain_order=True)
        .sort('__row', maintain_order=True)
    )
