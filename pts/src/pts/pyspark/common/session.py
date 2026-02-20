from __future__ import annotations

import os
from typing import TYPE_CHECKING

from loguru import logger
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession

if TYPE_CHECKING:
    from pyspark.sql import DataFrame
    from pyspark.sql.types import StructType


class Session:
    """This class provides a Spark session."""

    def __init__(
        self,
        app_name: str = 'pts',
        spark_uri: str = 'local[*]',
        properties: dict[str, str] | None = None,
    ) -> None:
        """Initializes a Spark Session."""
        self.is_dataproc = 'DATAPROC_CLUSTER_NAME' in os.environ

        self.spark: SparkSession = (
            SparkSession
            .Builder()
            .config(conf=self._create_config(properties))
            .master('yarn' if self.is_dataproc else spark_uri)
            .appName(app_name)
            .getOrCreate()
        )
        self.spark.sparkContext.setLogLevel('WARN')

    def _create_config(self, properties: dict[str, str] | None = None) -> SparkConf:
        if properties is None:
            properties = {}
        base_properties = {}

        if not self.is_dataproc:
            base_properties = {
                'spark.driver.maxResultSize': '0',
                'spark.debug.maxToStringFields': '2000',
                'spark.sql.broadcastTimeout': '3000',
                # google cloud storage connector
                'spark.jars.packages': 'com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.21',
                'spark.sql.adaptive.enabled': 'true',
                'spark.sql.adaptive.coalescePartitions.enabled': 'true',
                'spark.serializer': 'org.apache.spark.serializer.KryoSerializer',
                'spark.network.timeout': '10s',
                'spark.network.timeoutInterval': '10s',
                'spark.executor.heartbeatInterval': '6s',
                'spark.hadoop.fs.gs.block.size': '134217728',
                'spark.hadoop.fs.gs.inputstream.buffer.size': '8388608',
                'spark.hadoop.fs.gs.outputstream.buffer.size': '8388608',
                'spark.hadoop.fs.gs.outputstream.sync.min.interval.ms': '2000',
                'spark.hadoop.fs.gs.status.parallel.enable': 'true',
                'spark.hadoop.fs.gs.glob.algorithm': 'CONCURRENT',
                'spark.hadoop.fs.gs.copy.with.rewrite.enable': 'true',
                'spark.hadoop.fs.gs.metadata.cache.enable': 'false',
                'spark.hadoop.fs.gs.auth.type': 'APPLICATION_DEFAULT',
                'spark.hadoop.fs.gs.impl': 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem',
                'spark.hadoop.fs.AbstractFileSystem.gs.impl': 'com.google.cloud.hadoop.fs.gcs.GoogleHadoopFS',
            }

        effective_properties = {**base_properties, **properties}

        return SparkConf().setAll(list(effective_properties.items()))

    def load_data(
        self,
        path: str | list[str],
        format: str = 'parquet',
        schema: StructType | str | None = None,
        **kwargs: bool | float | int | str | None,
    ) -> DataFrame:
        """Generic function to read a file or folder into a Spark dataframe.

        The `recursiveFileLookup` flag when set to True will skip all partition
        columns, but read files from all subdirectories.

        Args:
            path (str | list[str]): path to the dataset
            format (str): file format. Defaults to parquet.
            schema (StructType | str | None): Schema to use when reading the data.
            **kwargs (bool | float | int | str | None): Additional arguments to
                pass to spark.read.load. `mergeSchema` is set to True,
                `recursiveFileLookup` is set to False by default.

        Returns:
            DataFrame: Dataframe
        """
        if schema is None:
            kwargs['inferSchema'] = kwargs.get('inferSchema', True)
        kwargs['mergeSchema'] = kwargs.get('mergeSchema', True)
        kwargs['recursiveFileLookup'] = kwargs.get('recursiveFileLookup', False)

        return self.spark.read.load(path, format=format, schema=schema, **kwargs)

    def stop(self) -> None:
        """Stops the Spark session."""
        self.spark.stop()
        logger.info('spark session stopped')
