"""Polars helpers shared by the IMPC evidence and mouse phenotype steps."""

from __future__ import annotations

from typing import Any

import polars as pl

from pts.schemas.impc import MGI_PUBMED_COLUMNS
from pts.transformers.utils.spark_csv import read_spark_csv

#: Working column: the model id.
MODEL_ID = 'model_id'

#: Java's `$` also matches before a final line terminator.
_MGI_MODEL_RE = r'^MGI:[0-9]+(?:\r\n|[\n\r\x{0085}\x{2028}\x{2029}])?$'


def biological_model_id(model_id: pl.Expr) -> pl.Expr:
    """The model id before its first `#`, if it is an MGI id; null otherwise."""
    cleaned = model_id.str.split('#').list.first()
    return pl.when(cleaned.str.contains(_MGI_MODEL_RE)).then(cleaned)


def explode_non_empty(lf: pl.LazyFrame, column: str) -> pl.LazyFrame:
    """Spark's `explode`: rows with a null or empty list are dropped."""
    return lf.filter(pl.col(column).list.len() > 0).explode(column)


def phenotype_ids(lf: pl.LazyFrame, column: str, prefix: str, id_column: str) -> pl.LazyFrame:
    """One row per `<prefix>:<digits>` id in `column`; rows without one are dropped."""
    return explode_non_empty(
        lf.with_columns(pl.col(column).str.extract_all(f'{prefix}:[0-9]+').alias(id_column)), id_column
    )


def ontology_terms(ontology: pl.LazyFrame, name: str) -> pl.LazyFrame:
    """The ids and terms of one ontology, as `<name>_id` and `<name>_term` in lower case."""
    prefix = name.lower()
    return ontology.filter(pl.col('ontology') == name).select(
        pl.col('phenotype_id').alias(f'{prefix}_id'), pl.col('phenotype_term').alias(f'{prefix}_term')
    )


def gene_mapping(mgi: pl.LazyFrame, mouse_to_human: pl.LazyFrame, hgnc: pl.LazyFrame) -> pl.LazyFrame:
    """MGI gene id to mouse symbol, mouse Ensembl id and human Ensembl id, through HGNC."""
    return (
        mgi
        .select(
            targetInModelMgiId=pl.col('1. MGI accession id'),
            targetInModel=pl.col('3. marker symbol'),
            targetInModelEnsemblId=pl.col('11. Ensembl gene id'),
        )
        .filter(pl.col('targetInModelEnsemblId').is_not_null())
        .join(
            mouse_to_human.select(targetInModelMgiId=pl.col('gene_id'), hgnc_gene_id=pl.col('hgnc_gene_id')),
            on='targetInModelMgiId',
            how='inner',
            maintain_order='left',
        )
        .join(
            hgnc.select(hgnc_gene_id=pl.col('hgnc_id'), targetFromSourceId=pl.col('ensembl_gene_id')),
            on='hgnc_gene_id',
            how='inner',
            maintain_order='left',
        )
        .filter(pl.col('targetFromSourceId').is_not_null())
        .select('targetInModelMgiId', 'targetInModel', 'targetInModelEnsemblId', 'targetFromSourceId')
    )


def literature(mgi_pubmed: pl.LazyFrame, summary: pl.LazyFrame, model_mouse: pl.LazyFrame) -> pl.LazyFrame:
    """Sorted distinct PMIDs per model and MGI gene, from the gene's references for the model's phenotypes."""
    references = mgi_pubmed.with_columns(pl.col('literature', 'targetInModelMgiId').str.split('|'))
    references = explode_non_empty(explode_non_empty(references, 'literature'), 'targetInModelMgiId')
    return (
        summary
        .select(MODEL_ID, 'targetInModelMgiId')
        .unique(maintain_order=True)
        .join(model_mouse, on=MODEL_ID, how='inner', maintain_order='left')
        .join(
            references.select('mp_id', 'literature', 'targetInModelMgiId').unique(maintain_order=True),
            on=['targetInModelMgiId', 'mp_id'],
            how='inner',
            maintain_order='left',
        )
        .group_by(MODEL_ID, 'targetInModelMgiId')
        .agg(pl.col('literature').unique().sort())
    )


def read_impc_csv(path: str, **options: Any) -> pl.LazyFrame:
    """A CSV read as Spark reads it, returned as a LazyFrame; `options` go to `read_spark_csv`."""
    return read_spark_csv(path, **options).lazy()


def read_gene_mapping(source: dict[str, str]) -> pl.LazyFrame:
    """`gene_mapping` of the `mouse_gene_mappings`, `solr_gene_gene` and `hgnc_gene_mappings` sources."""
    return gene_mapping(
        read_impc_csv(source['mouse_gene_mappings'], separator='\t', null_values=['null']),
        read_impc_csv(source['solr_gene_gene']),
        read_impc_csv(source['hgnc_gene_mappings'], separator='\t', null_values=['null']),
    )


def read_mgi_pubmed(path: str) -> pl.LazyFrame:
    """The headerless MGI PubMed references, named by `MGI_PUBMED_COLUMNS`."""
    return read_impc_csv(
        path,
        separator='\t',
        has_header=False,
        new_columns=MGI_PUBMED_COLUMNS,
        null_values=['null'],
    )


def model_mouse_phenotypes(path: str) -> pl.LazyFrame:
    """One row per model and `MP` phenotype id of the mouse model file, as `MODEL_ID` and `mp_id`."""
    return phenotype_ids(read_impc_csv(path), 'model_phenotypes', 'MP', 'mp_id').select(MODEL_ID, 'mp_id')
