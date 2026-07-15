"""Reader for the CFPB Consumer Complaint Database export (~3 GB CSV).

The narrative column contains free text with embedded commas, quotes and
newlines, so the reader must be told about quoting and multiline records.
Without these options Spark would split one complaint across several
broken rows.

Verify with:
    python -m src.ingestion.load_complaints
"""
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F

from src.ingestion.schemas import COMPLAINTS

RAW_PATH = "data/raw/cfpb_complaints/complaints.csv"


def load_complaints(spark: SparkSession) -> DataFrame:
    df = spark.read.csv(
        RAW_PATH,
        header=True,
        schema=COMPLAINTS,
        quote='"',
        escape='"',        # CSV escapes quotes by doubling them: ""
        multiLine=True,    # narratives contain literal newlines
        mode="PERMISSIVE", # tolerate stray malformed rows in a 3 GB export
    )
    # Normalize the raw export's column names to snake_case once, here,
    # so downstream code never deals with spaces or '?' in names.
    renames = {
        "Date received": "date_received",
        "Product": "product",
        "Sub-product": "sub_product",
        "Issue": "issue",
        "Sub-issue": "sub_issue",
        "Consumer complaint narrative": "narrative",
        "Company public response": "company_public_response",
        "Company": "company",
        "State": "state",
        "ZIP code": "zip_code",
        "Tags": "tags",
        "Submitted via": "submitted_via",
        "Date sent to company": "date_sent_to_company",
        "Company response to consumer": "company_response",
        "Timely response?": "timely_response",
        "Complaint ID": "complaint_id",
    }
    for old, new in renames.items():
        df = df.withColumnRenamed(old, new)
    # dates come as ISO strings like 2023-07-11T15:12:42.000Z;
    # try_to_timestamp gives null on the occasional malformed value
    # instead of failing the whole job under ANSI mode
    return df.withColumn(
        "date_received", F.to_date(F.try_to_timestamp("date_received"))
    )


def main() -> None:
    from src.common.spark import get_spark

    spark = get_spark("complaints-ingestion-check")
    df = load_complaints(spark)
    df.printSchema()
    total = df.count()
    with_narrative = df.filter(F.col("narrative").isNotNull()).count()
    print(f"total complaints: {total:,}")
    print(f"with narrative:   {with_narrative:,}")
    spark.stop()


if __name__ == "__main__":
    main()
