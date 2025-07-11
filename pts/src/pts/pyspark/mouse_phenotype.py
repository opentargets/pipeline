from loguru import logger

from pts.pyspark.common.session import Session


def mouse_phenotype(
    source: dict[str, str],
    destination: dict[str, str],
    properties: dict[str, str] | None,
) -> None:
    # start spark session
    spark = Session(app_name='mouse_phenotype', properties=properties)

    # load data from source paths
    logger.debug(f'loading data from: {source}')
    df = spark.load_data(source['mouse_phenotype'], format='json')
    target_df = spark.load_data(source['target'], format='parquet')

    logger.debug('performing left semi join to filter data')
    out_df = df.join(target_df, target_df['id'] == df['targetFromSourceId'], 'left_semi')
    logger.debug('performing left anti join to filter out data not in target_df')
    exc_df = df.join(out_df.select('targetFromSourceId'), ['targetFromSourceId'], 'left_anti')

    # write output data
    logger.debug(f'writing output data to: {destination}')
    out_df.write.parquet(destination['output'])
    exc_df.write.parquet(destination['excluded'])
