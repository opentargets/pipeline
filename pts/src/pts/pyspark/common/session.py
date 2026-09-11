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

    @staticmethod
    def _merge_jars_packages(base: str | None, extra: str | None) -> str | None:
        """Merge two ``spark.jars.packages`` comma-lists, deduping preserving order."""
        if not base and not extra:
            return None
        seen: set[str] = set()
        merged: list[str] = []
        for part in ((base or '') + ',' + (extra or '')).split(','):
            p = part.strip()
            if p and p not in seen:
                seen.add(p)
                merged.append(p)
        return ','.join(merged) if merged else None

    def _effective_properties(self, properties: dict[str, str] | None = None) -> dict[str, str]:
        """Return the merged Spark properties without touching JVM global state.

        Separated for unit testability: ``SparkConf`` inherits JVM/system
        properties once a ``SparkContext`` exists, so ``conf.get`` is
        polluted by previous ``Session``s in the same pytest session.
        Tests should assert on this dict, not on ``SparkConf.get``.
        """
        if properties is None:
            properties = {}
        base_properties: dict[str, str] = {}

        if not self.is_dataproc:
            base_properties = {
                'spark.driver.maxResultSize': '0',
                'spark.debug.maxToStringFields': '2000',
                'spark.sql.broadcastTimeout': '3000',
                # google cloud storage connector + Spark NLP (required by OnToma for local runs).
                # On Dataproc the jar is provided via ``spark.jars``
                # so base is empty there; locally we need Ivy resolution.
                'spark.jars.packages': ','.join([
                    'com.google.cloud.bigdataoss:gcs-connector:hadoop3-2.2.21',
                    'com.johnsnowlabs.nlp:spark-nlp_2.12:6.1.5',
                ]),
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
                'spark.sql.parquet.compression.codec': 'zstd',
            }

        # ``spark.jars.packages`` needs merging, not last-write-wins, so caller-supplied
        # jars are not dropped and duplicates are avoided.
        jars_key = 'spark.jars.packages'
        if jars_key in base_properties or jars_key in properties:
            merged = self._merge_jars_packages(
                base_properties.get(jars_key), properties.get(jars_key)
            )
            effective_properties = {**base_properties, **properties}
            if merged:
                effective_properties[jars_key] = merged
            else:
                effective_properties.pop(jars_key, None)
        else:
            effective_properties = {**base_properties, **properties}

        return effective_properties  # ty: ignore[invalid-return-type]

    def _create_config(self, properties: dict[str, str] | None = None) -> SparkConf:
        effective = self._effective_properties(properties)
        return SparkConf().setAll(list(effective.items()))

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
