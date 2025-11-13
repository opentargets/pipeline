"""Mouse phenotype dataset generation and filtering."""

from typing import Any

import pronto
import pyspark.sql.functions as f
from loguru import logger
from pyspark.sql import DataFrame

from pts.pyspark.common.impc import (
    build_gene_mapping,
    cleanup_model_identifier,
    format_disease_model_associations,
    process_literature_references,
    process_ontology_terms,
)
from pts.pyspark.common.session import Session


def mouse_phenotype(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Generate mouse phenotypes dataset from IMPC data and filter by target."""
    # start spark session
    spark = Session(app_name='mouse_phenotype', properties=properties)

    # Generate mouse phenotypes from IMPC source data
    logger.info('generating mouse phenotypes from IMPC source data')
    # Load IMPC datasets (mouse phenotypes subset)
    datasets = _load_impc_datasets_for_mouse_phenotypes(spark, source)
    datasets = _prepare_impc_datasets_for_mouse_phenotypes(datasets)

    # Process ontology terms and classifications (need spark instance)
    datasets['mp_class'] = _create_mp_classification(datasets['mp_ontology'], spark)

    # Generate mouse phenotypes dataset
    mouse_phenotypes_df = generate_mouse_phenotypes_dataset(
        datasets['disease_model_summary_transformed'],
        datasets['gene_mapping'],
        datasets['model_mouse_phenotypes_transformed'],
        datasets['mp_terms'],
        datasets['literature'],
        datasets['mp_class'],
    )

    logger.info('loading target data for filtering')
    target_df = spark.load_data(source['target'], format='parquet')
    out_df, exc_df = filter_mouse_phenotypes_by_target(mouse_phenotypes_df, target_df)

    # write output data
    logger.info(f'writing output data to: {destination}')
    out_df.write.mode('overwrite').parquet(destination['output'])
    exc_df.write.mode('overwrite').parquet(destination['excluded'])


def generate_mouse_phenotypes_dataset(
    disease_model_summary: DataFrame,
    gene_mapping: DataFrame,
    model_mouse_phenotypes: DataFrame,
    mp_terms: DataFrame,
    literature: DataFrame,
    mp_class: DataFrame,
) -> DataFrame:
    """Generate the related mousePhenotypes dataset for the corresponding widget in the target object."""
    logger.info('generate mouse phenotypes dataset')

    mouse_phenotypes = (
        # Extract base model-target associations.
        disease_model_summary.select(
            'model_id', 'biologicalModelAllelicComposition', 'biologicalModelGeneticBackground', 'targetInModelMgiId'
        )
        .distinct()
        # Add gene mapping information.
        .join(gene_mapping, on='targetInModelMgiId', how='inner')
        # Add mouse phenotypes.
        .join(model_mouse_phenotypes, on='model_id', how='inner')
        .join(mp_terms, on='mp_id', how='inner')
        # Add literature references.
        .join(literature, on=['model_id', 'targetInModelMgiId'], how='left')
        # Rename fields.
        .withColumnRenamed('mp_id', 'modelPhenotypeId')
        .withColumnRenamed('mp_term', 'modelPhenotypeLabel')
        # Join phenotype class information.
        .join(mp_class, on='modelPhenotypeId', how='inner')
        # Post-process model ID field.
        .withColumn('biologicalModelId', cleanup_model_identifier(f.col('model_id')))
    )

    return (
        # Convert the schema from flat to partially nested, grouping related models and phenotype classes.
        mouse_phenotypes.groupby(
            'targetInModel',
            'targetInModelMgiId',
            'targetInModelEnsemblId',
            'targetFromSourceId',
            'modelPhenotypeId',
            'modelPhenotypeLabel',
        ).agg(
            f.collect_set(
                f.struct(
                    f.col('biologicalModelAllelicComposition').alias('allelicComposition'),
                    f.col('biologicalModelGeneticBackground').alias('geneticBackground'),
                    f.col('biologicalModelId').alias('id'),
                    f.col('literature'),
                )
            ).alias('biologicalModels'),
            f.collect_set(
                f.struct(f.col('modelPhenotypeClassId').alias('id'), f.col('modelPhenotypeClassLabel').alias('label'))
            ).alias('modelPhenotypeClasses'),
        )
    )


def filter_mouse_phenotypes_by_target(
    mouse_phenotypes_df: DataFrame, target_df: DataFrame
) -> tuple[DataFrame, DataFrame]:
    """Filter mouse phenotypes based on target data and return both output and excluded data."""
    logger.info('performing left semi join to filter mouse phenotypes by target')
    out_df = mouse_phenotypes_df.join(
        target_df, target_df['id'] == mouse_phenotypes_df['targetFromSourceId'], 'left_semi'
    )

    logger.info('performing left anti join to identify excluded mouse phenotypes')
    exc_df = mouse_phenotypes_df.join(out_df.select('targetFromSourceId'), ['targetFromSourceId'], 'left_anti')

    return out_df, exc_df


def _load_impc_datasets_for_mouse_phenotypes(spark: Session, source: dict[str, str]) -> dict[str, DataFrame]:
    """Load IMPC datasets required for mouse phenotypes generation (subset of evidence data)."""
    logger.info(f'load IMPC data for mouse phenotypes generation from {source}')
    return {
        # Core datasets needed for mouse phenotypes
        'model_mouse_phenotypes': spark.load_data(source['solr_mouse_model'], format='csv', header=True),
        'disease_model_summary': spark.load_data(source['solr_disease_model_summary'], format='csv', header=True),
        'ontology': spark.load_data(source['solr_ontology'], format='csv', header=True),
        'mp_ontology': pronto.Ontology(source['mp_ontology']),
        'mgi_pubmed': spark.load_data(
            source['mouse_pubmed_refs'],
            format='csv',
            schema='_0 string, _1 string, _2 string, mp_id string, literature string, targetInModelMgiId string',
            nullValue='null',
            sep='\\t',
        ),
        'mgi_gene_id_to_ensembl_mouse_gene_id': spark.load_data(
            source['mouse_gene_mappings'], format='csv', header=True, nullValue='null', sep='\\t'
        ),
        'mouse_to_human_gene': spark.load_data(source['solr_gene_gene'], format='csv', header=True),
        'hgnc_gene_id_to_ensembl_human_gene_id': spark.load_data(
            source['hgnc_gene_mappings'], format='csv', header=True, nullValue='null', sep='\\t'
        ),
    }


def _prepare_impc_datasets_for_mouse_phenotypes(datasets: dict[str, DataFrame]) -> dict[str, DataFrame]:
    """Prepare transformed versions of IMPC datasets for mouse phenotypes generation."""
    # Transform model phenotypes
    datasets['model_mouse_phenotypes_transformed'] = (
        datasets['model_mouse_phenotypes']
        .withColumn('mp_id', f.explode(f.expr(r"regexp_extract_all(model_phenotypes, '(MP:\\d+)', 1)")))
        .select('model_id', 'mp_id')
        .persist()
    )

    # Transform disease model summary
    datasets['disease_model_summary_transformed'] = format_disease_model_associations(datasets['disease_model_summary'])

    # Process ontology terms
    datasets['mp_terms'], datasets['hp_terms'] = process_ontology_terms(datasets['ontology'])

    # Build gene mapping
    datasets['gene_mapping'] = build_gene_mapping(
        datasets['mgi_gene_id_to_ensembl_mouse_gene_id'],
        datasets['mouse_to_human_gene'],
        datasets['hgnc_gene_id_to_ensembl_human_gene_id'],
    )

    # Process literature references
    datasets['literature'] = process_literature_references(
        datasets['mgi_pubmed'],
        datasets['disease_model_summary_transformed'],
        datasets['model_mouse_phenotypes_transformed'],
    )

    return datasets


def _create_mp_classification(mp_ontology, spark: Session) -> DataFrame:
    """Process MP definitions to extract high level classes for each term."""
    logger.info('process MP definitions to extract high level classes for each term.')
    high_level_classes = set(mp_ontology['MP:0000001'].subclasses(distance=1)) - {mp_ontology['MP:0000001']}
    mp_class_data = [
        [term.id, mp_high_level_class.id, mp_high_level_class.name]
        for mp_high_level_class in high_level_classes
        for term in mp_high_level_class.subclasses()
    ]
    return spark.spark.createDataFrame(
        data=mp_class_data,
        schema=['modelPhenotypeId', 'modelPhenotypeClassId', 'modelPhenotypeClassLabel'],
    )
