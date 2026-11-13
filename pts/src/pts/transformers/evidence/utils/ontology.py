"""Map evidence diseases to EFO with OnToma lookup tables.

Polars port of the removed pyspark `add_efo_mapping`.
"""

from __future__ import annotations

import polars as pl
import polars.selectors as cs
from loguru import logger

from pts.transformers.utils.disease_mapping import map_entities, read_entity_lut
from pts.transformers.utils.disease_mapping.mapping import non_acronym_ids, temporary_prefix

MAPPED_ID_COLUMN = 'diseaseFromSourceMappedId'


def add_efo_mapping(
    evidence: pl.LazyFrame,
    disease_label_lut_path: str,
    label_col_name: str = 'diseaseFromSource',
    disease_id_lut_path: str | None = None,
    id_col_name: str | None = 'diseaseFromSourceId',
    *,
    keep_acronyms: bool = False,
    spelling_variants: bool = False,
) -> pl.LazyFrame:
    """Add `diseaseFromSourceMappedId`, the disease the evidence maps to, one row per mapping.

    The disease label is mapped with the label lookup table and, when both an id column and an id
    lookup table are given, the disease id with the id table too; the two sets of mappings are
    merged. Evidence with several mappings is exploded into one row each, and evidence without any
    is kept with a null mapping. Label mappings flagged as acronyms in the lookup table are dropped
    unless `keep_acronyms` is set. With `spelling_variants`, evidence that neither its label nor its
    id maps is retried with the label in American spelling and without bracketed single tokens.
    Identical evidence rows come out once.

    Args:
        evidence: Evidence holding the disease label and, optionally, id columns.
        disease_label_lut_path: OnToma disease label lookup table.
        label_col_name: Column holding the disease label.
        disease_id_lut_path: OnToma disease id lookup table.
        id_col_name: Column holding the disease id.
        keep_acronyms: Keep label mappings flagged as acronyms.
        spelling_variants: Retry the label of evidence left without any mapping
            (`map_entities`' `spelling_variants`).

    Returns:
        The evidence with `diseaseFromSourceMappedId` added.
    """
    # the per-lookup results are working columns, kept clear of every evidence column
    prefix = temporary_prefix(evidence.collect_schema().names())
    label_result, id_result, retry_result = f'{prefix}label', f'{prefix}id', f'{prefix}retry'

    logger.info(f'load disease label lookup table from {disease_label_lut_path}')
    mapped = map_entities(
        evidence,
        read_entity_lut(disease_label_lut_path),
        label_result,
        label_col_name,
        'label',
        type_col=pl.lit('DS'),
        include_entity_source=True,
    )
    all_ids = pl.col(label_result).list.eval(pl.element().struct.field('entityId'))
    if keep_acronyms:
        mapped = mapped.with_columns(all_ids.list.unique().list.sort().alias(label_result))
    else:
        kept_ids = non_acronym_ids(pl.col(label_result))
        labelled = mapped.collect()
        lost = labelled.select((all_ids.list.n_unique() > kept_ids.list.len().fill_null(0)).sum()).item()
        logger.info(f'{lost} evidence rows lost disease ids from acronym label mappings')
        mapped = labelled.lazy().with_columns(kept_ids.alias(label_result))
    mapped_ids = pl.col(label_result).fill_null([])

    if id_col_name and disease_id_lut_path:
        logger.info(f'load disease id lookup table from {disease_id_lut_path}')
        mapped = map_entities(
            mapped,
            read_entity_lut(disease_id_lut_path),
            id_result,
            id_col_name,
            'id',
            type_col=pl.lit('DS'),
        )
        mapped_ids = pl.concat_list(mapped_ids, pl.col(id_result).fill_null([])).list.unique().list.sort()

    if spelling_variants:
        mapped = map_entities(
            mapped,
            read_entity_lut(disease_label_lut_path),
            retry_result,
            label_col_name,
            'label',
            type_col=pl.lit('DS'),
            include_entity_source=True,
            spelling_variants=True,
        )
        retry_structs = pl.col(retry_result)
        retry_ids = (
            retry_structs.list.eval(pl.element().struct.field('entityId')).list.unique().list.sort()
            if keep_acronyms
            else non_acronym_ids(retry_structs)
        )
        mapped_ids = pl.when(mapped_ids.list.len() > 0).then(mapped_ids).otherwise(retry_ids.fill_null([]))

    return (
        mapped
        .with_columns(mapped_ids.alias(MAPPED_ID_COLUMN))
        .drop(cs.starts_with(prefix))
        # explode_outer: no mapping still leaves one row, with a null id
        .explode(MAPPED_ID_COLUMN, empty_as_null=True)
    )
