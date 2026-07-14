"""Readers for the TheLook eCommerce raw CSVs.

Each function returns a lazy DataFrame: nothing is read from disk until
an action (count/show/write) runs downstream. `load_us_users` is the
scoping step for the whole project: TheLook is a global dataset, and the
churn analysis is deliberately limited to US customers so the state-level
comparison is apples-to-apples.

Verify the raw files load correctly with:
    python -m src.ingestion.load_thelook
"""
from pyspark.sql import DataFrame, SparkSession

from src.ingestion.schemas import THELOOK_SCHEMAS

RAW_DIR = "data/raw/thelook_ecommerce"

US_COUNTRY_VALUE = "United States"  # exact string used in users.country


def load_table(spark: SparkSession, name: str) -> DataFrame:
    df = spark.read.csv(
        f"{RAW_DIR}/{name}.csv",
        header=True,
        schema=THELOOK_SCHEMAS[name],
        # Fail the read if a row doesn't match the schema instead of
        # silently nulling fields (default PERMISSIVE mode would hide
        # data problems until much later).
        mode="FAILFAST",
    )
    if name == "events":
        # undo the pandas float export artifact, see schemas.py
        df = df.withColumn("user_id", df["user_id"].cast("long"))
    return df


def load_us_users(spark: SparkSession) -> DataFrame:
    return load_table(spark, "users").filter(
        f"country = '{US_COUNTRY_VALUE}'"
    )


def main() -> None:
    from src.common.spark import get_spark

    spark = get_spark("thelook-ingestion-check")
    for name in THELOOK_SCHEMAS:
        df = load_table(spark, name)
        print(f"{name}: {df.count():,} rows")
    us = load_us_users(spark)
    print(f"US users: {us.count():,}")
    spark.stop()


if __name__ == "__main__":
    main()
