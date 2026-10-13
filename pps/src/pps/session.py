"""Spark session shared by the steps of a pps task."""

from __future__ import annotations

import os
from typing import Literal

from pydantic import BaseModel
from pyspark.sql import SparkSession

WriteMode = Literal['overwrite', 'append', 'ignore', 'error', 'errorifexists']


class SessionSettings(BaseModel):
    """Spark session settings, set per task under `session` in `config.yaml`."""

    spark_uri: str | None = None
    """Spark master. When unset, the master provided by spark-submit is used (e.g.
    yarn on Dataproc), falling back to `local[*]` when running outside spark-submit."""
    app_name: str = 'pps'
    """Spark application name."""
    write_mode: WriteMode = 'errorifexists'
    """Save mode used by the steps when writing their outputs."""
    output_partitions: int = 200
    """Number of partitions the steps should coalesce their outputs to."""
    extended_spark_conf: dict[str, str] = {}
    """Extra spark configuration applied to the session."""
    log_level: str = 'ERROR'
    """Spark log level."""


class Session:
    """Spark session wrapper handed to every step function."""

    def __init__(self, settings: SessionSettings | None = None) -> None:
        self.settings = settings or SessionSettings()

        builder = SparkSession.builder.appName(self.settings.app_name)
        for key, value in self.settings.extended_spark_conf.items():
            builder = builder.config(key, value)
        if master := self._master():
            builder = builder.master(master)

        self.spark = builder.getOrCreate()
        self.spark.sparkContext.setLogLevel(self.settings.log_level)

    def _master(self) -> str | None:
        if self.settings.spark_uri:
            return self.settings.spark_uri
        # spark-submit (e.g. a Dataproc job) launches the driver with a gateway
        # and its own master, which must not be overridden
        if 'PYSPARK_GATEWAY_PORT' in os.environ:
            return None
        return 'local[*]'

    @property
    def write_mode(self) -> WriteMode:
        """Save mode used by the steps when writing their outputs."""
        return self.settings.write_mode

    @property
    def output_partitions(self) -> int:
        """Number of partitions the steps should coalesce their outputs to."""
        return self.settings.output_partitions
