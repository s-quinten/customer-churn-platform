"""End-to-end build of the churn training dataset.

Reads the raw TheLook CSVs, derives the churn label with a temporal
cutoff, computes per-customer features as of that cutoff, and writes the
result to data/processed/churn_dataset as parquet.

Run:
    python -m src.processing.build_dataset
"""
from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.ingestion.load_thelook import load_table, load_us_users
from src.processing.churn_label import CHURN_WINDOW_DAYS, cutoff_date, label_churn
from src.processing.features import build_feature_table

OUTPUT_PATH = "data/processed/churn_dataset"


def main() -> None:
    spark = get_spark("build-churn-dataset")
    spark.sparkContext.setLogLevel("ERROR")

    users = load_us_users(spark)
    orders = load_table(spark, "orders")
    order_items = load_table(spark, "order_items")
    products = load_table(spark, "products")
    events = load_table(spark, "events")

    cutoff = cutoff_date(orders, CHURN_WINDOW_DAYS)
    print(f"cutoff date: {cutoff} (window: {CHURN_WINDOW_DAYS} days)")

    labels = label_churn(orders, cutoff)
    dataset = build_feature_table(
        users, orders, order_items, products, events, labels, cutoff
    )

    dataset.write.mode("overwrite").parquet(OUTPUT_PATH)

    # Read back what was written: verifies the files and the numbers
    # in one go.
    written = spark.read.parquet(OUTPUT_PATH)
    total = written.count()
    churned = written.filter(F.col("churned") == 1).count()
    print(f"rows written: {total:,}")
    print(f"churned: {churned:,} ({churned / total:.1%})")
    print(f"columns: {len(written.columns)}")
    written.show(5)
    spark.stop()


if __name__ == "__main__":
    main()
