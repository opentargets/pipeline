"""Common functions for IMPC data processing."""

import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import Column, DataFrame


def format_disease_model_associations(disease_model_summary: DataFrame) -> DataFrame:
    """Format disease model associations with proper column names."""
    return disease_model_summary.selectExpr(
        'model_id',
        'model_genetic_background as biologicalModelGeneticBackground',
        'model_description as biologicalModelAllelicComposition',
        'disease_id',
        'disease_term',
        'disease_model_avg_norm',
        'marker_id as targetInModelMgiId',
        'disease_model_avg_norm as resourceScore',
    ).distinct()


def process_ontology_terms(ontology: DataFrame) -> tuple[DataFrame, ...]:
    """Process ontology information to enable MP and HP term lookup based on the ID."""
    logger.info('process ontology information to enable MP and HP term lookup based on the ID.')
    return tuple(
        ontology.filter(f.col('ontology') == ontology_name)
        .withColumnRenamed('phenotype_id', f'{ontology_name.lower()}_id')
        .withColumnRenamed('phenotype_term', f'{ontology_name.lower()}_term')
        .select(f'{ontology_name.lower()}_id', f'{ontology_name.lower()}_term')
        for ontology_name in ('MP', 'HP')
    )


def build_gene_mapping(
    mgi_gene_id_to_ensembl_mouse_gene_id: DataFrame,
    mouse_to_human_gene: DataFrame,
    hgnc_gene_id_to_ensembl_human_gene_id: DataFrame,
) -> DataFrame:
    """Build complete gene mapping from MGI gene ID to human Ensembl ID."""
    logger.info('construct gene mapping.')
    return (
        mgi_gene_id_to_ensembl_mouse_gene_id.selectExpr(
            '`1. MGI accession id` as targetInModelMgiId',
            '`3. marker symbol` as targetInModel',
            '`11. Ensembl gene id` as targetInModelEnsemblId',
        )
        .filter(f.col('targetInModelEnsemblId').isNotNull())
        .join(
            mouse_to_human_gene.selectExpr('gene_id as targetInModelMgiId', 'hgnc_gene_id'),
            on='targetInModelMgiId',
            how='inner',
        )
        .join(
            hgnc_gene_id_to_ensembl_human_gene_id.selectExpr(
                'hgnc_id as hgnc_gene_id', 'ensembl_gene_id as targetFromSourceId'
            ),
            on='hgnc_gene_id',
            how='inner',
        )
        .filter(f.col('targetFromSourceId').isNotNull())
        .select('targetInModelMgiId', 'targetInModel', 'targetInModelEnsemblId', 'targetFromSourceId')
    )


def process_literature_references(
    mgi_pubmed: DataFrame, disease_model_summary: DataFrame, model_mouse_phenotypes: DataFrame
) -> DataFrame:
    """Process literature references for model-gene combinations."""
    mgi_pubmed_exploded = (
        mgi_pubmed.withColumn('literature', f.explode(f.split(f.col('literature'), r'\|')))
        .withColumn('targetInModelMgiId', f.explode(f.split(f.col('targetInModelMgiId'), r'\|')))
        .select('mp_id', 'literature', 'targetInModelMgiId')
        .distinct()
    )

    return (
        disease_model_summary.select('model_id', 'targetInModelMgiId')
        .distinct()
        .join(model_mouse_phenotypes, on='model_id', how='inner')
        .join(mgi_pubmed_exploded, on=['targetInModelMgiId', 'mp_id'], how='inner')
        .groupby('model_id', 'targetInModelMgiId')
        .agg(f.collect_set(f.col('literature')).alias('literature'))
        .select('model_id', 'targetInModelMgiId', 'literature')
    )


def cleanup_model_identifier(model_id_col: Column):
    """Cleanup model identifier by stripping modifiers and filtering MGI namespace."""
    cleaned_id = f.split(model_id_col, '#').getItem(0)
    return f.when(cleaned_id.rlike(r'^MGI:\d+$'), cleaned_id)
