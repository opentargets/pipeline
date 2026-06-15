from typing import Any

from loguru import logger

from pts.pyspark.common.session import Session
from pts.pyspark.expression_utils.run_cellex import CellexAnalysis

# Default Spark properties for the specificity step
_SPECIFICITY_DEFAULT_PROPERTIES: dict[str, str] = {
    'spark.driver.memory': '50g',
    'spark.executor.memory': '70g',
    'spark.memory.offHeap.enabled': 'true',
    'spark.memory.offHeap.size': '16g',
    'spark.driver.maxResultSize': '32g',
    'spark.sql.pivotMaxValues': '1000000',
}


def baseline_expression_specificity(
    source: str,
    destination: str,
    settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    logger.info('Starting baseline expression specificity computation')

    # Merge step defaults with any caller-supplied overrides
    effective_properties = {**_SPECIFICITY_DEFAULT_PROPERTIES, **properties}

    session = Session(app_name='baseline_expression_specificity', properties=effective_properties)
    spark = session.spark

    # Determine input path
    input_path = source

    specificity_method = settings.get('specificity_method', 'cellex')

    if specificity_method == 'cellex':
        # Extract settings
        biosample = settings['biosample']
        mode = settings.get('mode', 'parquet')

        cellex_analysis = CellexAnalysis(
            spark=spark,
            mode=mode,
            input_path=input_path,
            output_path=destination,
            biosample=biosample,
            sample_id=settings.get('sample_id'),
            do_anova=settings.get('do_anova', True),
        )
        cellex_analysis.run()
    else:
        raise ValueError(f"Specificity method '{specificity_method}' not supported.")
