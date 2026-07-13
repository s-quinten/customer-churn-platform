"""Explicit schemas for every raw input table.

Why not inferSchema=True? Two reasons:
1. Performance: schema inference forces Spark to read every file twice
   (once to guess types, once to load). On the 3 GB complaints file that
   doubles ingestion time for no benefit.
2. Safety: inference silently guesses. If a future download changes a
   column, an explicit schema fails loudly at read time instead of
   producing wrong types downstream.

Column names/types were verified against the actual files on 2026-07-13.
"""
from pyspark.sql.types import (
    DoubleType,
    IntegerType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

# ---------------------------------------------------------------------------
# TheLook eCommerce (structured backbone, churn features come from here)
# ---------------------------------------------------------------------------

USERS = StructType([
    StructField("id", LongType(), nullable=False),
    StructField("first_name", StringType()),
    StructField("last_name", StringType()),
    StructField("email", StringType()),
    StructField("age", IntegerType()),
    StructField("gender", StringType()),
    StructField("state", StringType()),
    StructField("street_address", StringType()),
    StructField("postal_code", StringType()),
    StructField("city", StringType()),
    StructField("country", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
    StructField("traffic_source", StringType()),
    StructField("created_at", TimestampType()),
])

ORDERS = StructType([
    StructField("order_id", LongType(), nullable=False),
    StructField("user_id", LongType()),
    StructField("status", StringType()),
    StructField("gender", StringType()),
    StructField("created_at", TimestampType()),
    StructField("returned_at", TimestampType()),
    StructField("shipped_at", TimestampType()),
    StructField("delivered_at", TimestampType()),
    StructField("num_of_item", IntegerType()),
])

ORDER_ITEMS = StructType([
    StructField("id", LongType(), nullable=False),
    StructField("order_id", LongType()),
    StructField("user_id", LongType()),
    StructField("product_id", LongType()),
    StructField("inventory_item_id", LongType()),
    StructField("status", StringType()),
    StructField("created_at", TimestampType()),
    StructField("shipped_at", TimestampType()),
    StructField("delivered_at", TimestampType()),
    StructField("returned_at", TimestampType()),
    StructField("sale_price", DoubleType()),
])

PRODUCTS = StructType([
    StructField("id", LongType(), nullable=False),
    StructField("cost", DoubleType()),
    StructField("category", StringType()),
    StructField("name", StringType()),
    StructField("brand", StringType()),
    StructField("retail_price", DoubleType()),
    StructField("department", StringType()),
    StructField("sku", StringType()),
    StructField("distribution_center_id", LongType()),
])

EVENTS = StructType([
    StructField("id", LongType(), nullable=False),
    StructField("user_id", LongType()),          # null for anonymous sessions
    StructField("sequence_number", IntegerType()),
    StructField("session_id", StringType()),
    StructField("created_at", TimestampType()),
    StructField("ip_address", StringType()),
    StructField("city", StringType()),
    StructField("state", StringType()),
    StructField("postal_code", StringType()),
    StructField("browser", StringType()),
    StructField("traffic_source", StringType()),
    StructField("uri", StringType()),
    StructField("event_type", StringType()),
])

DISTRIBUTION_CENTERS = StructType([
    StructField("id", LongType(), nullable=False),
    StructField("name", StringType()),
    StructField("latitude", DoubleType()),
    StructField("longitude", DoubleType()),
])

INVENTORY_ITEMS = StructType([
    StructField("id", LongType(), nullable=False),
    StructField("product_id", LongType()),
    StructField("created_at", TimestampType()),
    StructField("sold_at", TimestampType()),
    StructField("cost", DoubleType()),
    StructField("product_category", StringType()),
    StructField("product_name", StringType()),
    StructField("product_brand", StringType()),
    StructField("product_retail_price", DoubleType()),
    StructField("product_department", StringType()),
    StructField("product_sku", StringType()),
    StructField("product_distribution_center_id", LongType()),
])

THELOOK_SCHEMAS = {
    "users": USERS,
    "orders": ORDERS,
    "order_items": ORDER_ITEMS,
    "products": PRODUCTS,
    "events": EVENTS,
    "distribution_centers": DISTRIBUTION_CENTERS,
    "inventory_items": INVENTORY_ITEMS,
}

# ---------------------------------------------------------------------------
# CFPB Consumer Complaint Database (NLP layer)
# ---------------------------------------------------------------------------
# "Date received" / "Date sent to company" are dates without time; kept as
# StringType at ingestion and cast with to_date() in the cleaning step,
# because CFPB formats them as MM/DD/YY-style strings that Spark's CSV
# timestamp parser trips over.

COMPLAINTS = StructType([
    StructField("Date received", StringType()),
    StructField("Product", StringType()),
    StructField("Sub-product", StringType()),
    StructField("Issue", StringType()),
    StructField("Sub-issue", StringType()),
    StructField("Consumer complaint narrative", StringType()),
    StructField("Company public response", StringType()),
    StructField("Company", StringType()),
    StructField("State", StringType()),
    StructField("ZIP code", StringType()),
    StructField("Tags", StringType()),
    StructField("Submitted via", StringType()),
    StructField("Date sent to company", StringType()),
    StructField("Company response to consumer", StringType()),
    StructField("Timely response?", StringType()),
    StructField("Complaint ID", LongType()),
])
