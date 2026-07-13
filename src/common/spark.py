"""Shared SparkSession factory.

Every pipeline stage gets its session from here so the configuration
lives in exactly one place. Locally this runs Spark in embedded mode
(`local[*]` = use all CPU cores on this machine); on a cluster the
master URL comes from the environment instead, and no pipeline code
has to change.
"""
import os

from pyspark.sql import SparkSession


def get_spark(app_name: str) -> SparkSession:
    master = os.environ.get("SPARK_MASTER", "local[*]")
    return (
        SparkSession.builder
        .appName(app_name)
        .master(master)
        # Spark's RPC layer builds a spark:// URL from the machine's
        # hostname and rejects underscores in it (invalid URL chars).
        # Binding to localhost sidesteps hostname quirks entirely.
        # Correct for local mode; a cluster overrides this via env.
        .config("spark.driver.host", "localhost")
        .config("spark.driver.bindAddress", "127.0.0.1")
        # Hadoop's auth layer still calls Subject.getSubject, which JDK 23
        # forbids unless the (deprecated) security manager is re-allowed.
        # Needed for Java 23 hosts; harmless on Java 17/21.
        .config("spark.driver.extraJavaOptions", "-Djava.security.manager=allow")
        .config("spark.executor.extraJavaOptions", "-Djava.security.manager=allow")
        # Timestamps in the raw CSVs are UTC; pinning the session timezone
        # makes date arithmetic reproducible regardless of the host's locale.
        .config("spark.sql.session.timeZone", "UTC")
        .getOrCreate()
    )
