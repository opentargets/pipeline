"""Drug warnings as produced by ChEMBL.

Drug warnings are manually curated by ChEMBL according to the methodology outlined
in https://pubs.acs.org/doi/pdf/10.1021/acs.chemrestox.0c00296
"""

from typing import Any

from loguru import logger
from pyspark.sql import functions as f

from pts.pyspark.common.session import Session


def drug_warning(
    source: str,
    destination: str,
    _settings: dict[str, Any],
    properties: dict[str, str],
) -> None:
    """Transform ChEMBL drug warnings into the Open Targets format.

    Args:
        source: Path to the ChEMBL drug warnings JSONL file.
        destination: Path to write the output parquet file.
        _settings: Custom settings (not used).
        properties: Spark configuration options.
    """
    spark = Session(app_name='drug_warning', properties=properties)

    logger.info(f'Loading drug warnings from {source}')
    warnings_df = spark.load_data(source, format='json')

    logger.info('Preparing drug warnings')
    output_df = warnings_df.selectExpr(
        '_metadata.all_molecule_chembl_ids as chemblIds',
        'warning_class as toxicityClass',
        'warning_country as country',
        'warning_description as description',
        'warning_id as id',
        'warning_refs as references',
        'warning_type as warningType',
        'warning_year as year',
        'efo_term as efoTerm',
        'efo_id as efoId',
        'efo_id_for_warning_class as efoIdForWarningClass',
    ).withColumn(
        'references',
        f.transform(
            f.col('references'),
            lambda x: f.struct(
                x['ref_id'].alias('id'),
                x['ref_type'].alias('source'),
                x['ref_url'].alias('url'),
            ),
        ),
    )

    logger.info(f'Writing drug warnings to {destination}')
    output_df.write.parquet(destination, mode='overwrite')
