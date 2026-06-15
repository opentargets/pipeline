from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.expression_utils.dice import DiceBaselineExpression
from pts.pyspark.expression_utils.gtex import GtexBaselineExpression
from pts.pyspark.expression_utils.pride import PrideBaselineExpression
from pts.pyspark.expression_utils.pseudobulk_sc import PseudobulkExpression

# Default Spark properties per harmoniser type
_HARMONISE_LIGHT_PROPERTIES: dict[str, str] = {
    'spark.driver.memory': '8g',
    'spark.executor.memory': '8g',
}

_HARMONISE_HEAVY_PROPERTIES: dict[str, str] = {
    'spark.driver.memory': '50g',
    'spark.executor.memory': '70g',
    'spark.memory.offHeap.enabled': 'true',
    'spark.memory.offHeap.size': '16g',
    'spark.driver.maxResultSize': '32g',
    'spark.sql.pivotMaxValues': '1000000',
}

# Map harmoniser name → default properties
_HARMONISE_PROPERTIES: dict[str, dict[str, str]] = {
    'dice': _HARMONISE_LIGHT_PROPERTIES,
    'cellxgene': _HARMONISE_LIGHT_PROPERTIES,
    'pride': _HARMONISE_LIGHT_PROPERTIES,
    'gtex': _HARMONISE_HEAVY_PROPERTIES,
}


def baseline_expression_harmonise(
    source: dict[str, str],
    destination: dict[str, str],
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression computation')

    harmoniser = settings['harmoniser']

    # Merge step defaults with any caller-supplied overrides
    default_props = _HARMONISE_PROPERTIES.get(harmoniser, _HARMONISE_LIGHT_PROPERTIES)
    effective_properties = {**default_props, **properties}

    session = Session(app_name='baseline_expression_harmonise', properties=effective_properties)
    spark = session.spark

    if harmoniser == 'dice':
        # Extract arguments
        dice_directory = source['dice_directory']
        mapping_path = source['mapping_path']
        output_directory_path = destination['baseline_expression']

        # Run DICE processing
        DiceBaselineExpression(
            spark=spark,
            dice_directory=dice_directory,
            mapping_path=mapping_path,
            output_directory_path=output_directory_path,
            json=False,
            local=False,
        ).run()
    elif harmoniser == 'cellxgene':
        # Extract arguments
        h5ad_path = source['h5ad_path']
        output_directory_path = destination['baseline_expression']

        datasource_id = settings.get('datasource_id', 'tabula_sapiens')
        datatype_id = settings.get('datatype_id', 'scrna-seq')

        processor = PseudobulkExpression(
            spark=spark,
            h5ad_path=h5ad_path,
            output_directory_path=output_directory_path,
            datasource_id=datasource_id,
            datatype_id=datatype_id,
            json=False,
        )

        processor.run(
            min_cells=settings.get('min_cells', 5),
            min_genes=settings.get('min_genes', 5),
            technology=settings.get('technology', '10X'),
            normalise=settings.get('normalise', False),
            aggregation_min_cells=settings.get('aggregation_min_cells', 5),
            aggregation_method=settings.get('aggregation_method', 'sum'),
            tissue_agg_colname=settings.get('tissue_agg_colname', 'tissue_ontology_term_id'),
            celltype_agg_colname=settings.get('celltype_agg_colname', 'cell_type_ontology_term_id'),
            age_colname=settings.get('age_colname', 'age'),
            sex_colname=settings.get('sex_colname', 'sex'),
            ethnicity_colname=settings.get('ethnicity_colname', 'ethnicity'),
            donor_colname=settings.get('donor_colname', 'donor_id'),
        )

    elif harmoniser == 'pride':
        pride_source_data_dir = source['pride_directory']
        pride_codes = settings['pride_codes']
        tissue_ontology_mapping_path = source['tissue_ontology_mapping_path']
        output_directory_path = destination['baseline_expression']

        PrideBaselineExpression(
            spark=spark,
            pride_source_data_dir=pride_source_data_dir,
            pride_codes=pride_codes,
            output_directory_path=output_directory_path,
            tissue_ontology_mapping_path=tissue_ontology_mapping_path,
            json=False,
            local=False,
        ).run()

    elif harmoniser == 'gtex':
        gtex_source_data_path = source['gtex_source_data_path']
        sample_metadata_path = source['sample_metadata_path']
        subject_metadata_path = source['subject_metadata_path']
        output_directory_path = destination['baseline_expression']

        GtexBaselineExpression(
            spark=spark,
            gtex_source_data_path=gtex_source_data_path,
            output_directory_path=output_directory_path,
            sample_metadata_path=sample_metadata_path,
            subject_metadata_path=subject_metadata_path,
            json=False,
            local=False,
            matrix=settings.get('matrix', False),
        ).run()

    else:
        raise NotImplementedError(f'Harmoniser "{harmoniser}" is not implemented.')
