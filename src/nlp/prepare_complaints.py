"""Prepare the CFPB complaints for the NLP model and the dashboard.

Step 1 (slow, runs once): parse the 3 GB CSV and rewrite it as parquet.
The CSV has multiline narratives, so Spark can only read it with one
task. Parquet is splittable and columnar, so every step after this one
reads in parallel and only touches the columns it needs. This is the
usual raw-to-bronze conversion in a data platform.

Step 2 (fast, reruns freely): from the parquet, write
  - data/processed/complaints_sample: cleaned 10% sample of narratives
    with product labels, training data for the text classifier
  - data/processed/complaint_stats: complaint counts per state, product
    and month for the dashboard

Run inside the spark container:
    python -m src.nlp.prepare_complaints
"""
import os

from pyspark.sql import functions as F

from src.common.spark import get_spark
from src.ingestion.load_complaints import load_complaints

BRONZE_PATH = "data/processed/complaints_bronze"
SAMPLE_PATH = "data/processed/complaints_sample"
STATS_PATH = "data/processed/complaint_stats"

SAMPLE_FRACTION = 0.10  # ~230k narratives, plenty for TF-IDF training

US_STATES = (
    "AL AK AZ AR CA CO CT DE FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS "
    "MO MT NE NV NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV "
    "WI WY DC"
).split()


def ensure_bronze(spark) -> None:
    """One-time CSV to parquet conversion, skipped when already done."""
    if os.path.exists(os.path.join(BRONZE_PATH, "_SUCCESS")):
        print("bronze parquet already exists, skipping conversion")
        return
    print("converting csv to parquet (single task, takes a while)...")
    load_complaints(spark).write.mode("overwrite").parquet(BRONZE_PATH)
    print(f"wrote {BRONZE_PATH}")


def clean(df):
    """Keep usable rows, normalize the narrative text a little.

    The CFPB redacts personal data as XXXX blocks. Those tokens carry no
    meaning, so they get stripped before the text reaches the model.
    """
    return (
        df.filter(F.col("narrative").isNotNull())
        .filter(F.col("state").isin(US_STATES))
        .filter(F.col("product").isNotNull())
        .withColumn(
            "narrative",
            F.regexp_replace("narrative", r"X{2,}[\\d/]*", " "),
        )
        .withColumn("narrative", F.regexp_replace("narrative", r"\\s+", " "))
        .filter(F.length("narrative") >= 100)  # drop stubs with no content
        .dropDuplicates(["complaint_id"])
    )


def main() -> None:
    spark = get_spark("prepare-complaints")
    spark.sparkContext.setLogLevel("ERROR")

    ensure_bronze(spark)
    df = clean(spark.read.parquet(BRONZE_PATH))

    (
        df.sample(fraction=SAMPLE_FRACTION, seed=42)
        .select("complaint_id", "product", "state", "narrative")
        .write.mode("overwrite")
        .parquet(SAMPLE_PATH)
    )
    print(f"wrote sample to {SAMPLE_PATH}")

    stats = (
        df.groupBy(
            "state",
            "product",
            F.date_trunc("month", "date_received").alias("month"),
        )
        .agg(F.count("complaint_id").alias("complaint_count"))
    )
    stats.write.mode("overwrite").parquet(STATS_PATH)
    print(f"wrote stats to {STATS_PATH}")
    spark.stop()


if __name__ == "__main__":
    main()
