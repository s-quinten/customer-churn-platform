"""Per-customer feature engineering, all computed as of the cutoff date.

Every aggregate here only looks at rows with created_at <= cutoff. The
churn label looks at what happens after the cutoff, so keeping this rule
strict is what makes the model honest (see churn_label.py).

Feature groups:
  orders      -> recency, frequency, tenure, ordering rhythm
  order_items -> spend, basket size, product variety, return rate
  events      -> browsing intensity, sessions, last-seen recency
  users       -> demographics and acquisition channel (age, state, ...)

Categorical columns stay as plain strings here. Encoding them is a
modeling decision and belongs to the ML step, not the ETL step.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from src.processing.churn_label import valid_orders


def order_features(orders: DataFrame, cutoff) -> DataFrame:
    hist = valid_orders(orders).filter(F.col("created_at") <= F.lit(cutoff))
    return hist.groupBy("user_id").agg(
        F.count("order_id").alias("order_count"),
        F.datediff(F.lit(cutoff), F.max("created_at")).alias("recency_days"),
        F.datediff(F.lit(cutoff), F.min("created_at")).alias("tenure_days"),
        F.sum(
            (F.col("created_at") >= F.date_sub(F.lit(cutoff), 365)).cast("int")
        ).alias("orders_last_365d"),
        F.avg("num_of_item").alias("avg_items_per_order"),
    )


def spend_features(order_items: DataFrame, products: DataFrame, cutoff) -> DataFrame:
    hist = order_items.filter(
        (F.col("created_at") <= F.lit(cutoff))
        & (F.col("status") != "Cancelled")
    )
    with_products = hist.join(
        products.select(
            F.col("id").alias("product_id"), "category", "department"
        ),
        on="product_id",
        how="left",
    )
    return with_products.groupBy("user_id").agg(
        F.round(F.sum("sale_price"), 2).alias("total_spent"),
        F.round(F.avg("sale_price"), 2).alias("avg_item_price"),
        F.count("id").alias("items_bought"),
        F.countDistinct("category").alias("distinct_categories"),
        F.avg((F.col("status") == "Returned").cast("int")).alias("return_rate"),
    )


def event_features(events: DataFrame, cutoff) -> DataFrame:
    hist = events.filter(
        F.col("user_id").isNotNull()  # anonymous sessions can't be joined
        & (F.col("created_at") <= F.lit(cutoff))
    )
    return hist.groupBy("user_id").agg(
        F.countDistinct("session_id").alias("session_count"),
        (F.count("id") / F.countDistinct("session_id")).alias(
            "avg_events_per_session"
        ),
        F.datediff(F.lit(cutoff), F.max("created_at")).alias(
            "days_since_last_visit"
        ),
        F.sum((F.col("event_type") == "cart").cast("int")).alias("cart_events"),
    )


def user_features(users: DataFrame) -> DataFrame:
    # Demographics don't depend on the cutoff, they are fixed attributes.
    return users.select(
        F.col("id").alias("user_id"),
        "age",
        "gender",
        "state",
        "traffic_source",
    )


def build_feature_table(
    users: DataFrame,
    orders: DataFrame,
    order_items: DataFrame,
    products: DataFrame,
    events: DataFrame,
    labels: DataFrame,
    cutoff,
) -> DataFrame:
    """Join every feature group onto the labeled user set.

    Starts from `labels` (which defines who is in scope) and left joins
    the rest, so a user with orders but no browsing events still gets a
    row, with event columns null. Nulls are filled with 0 where "no
    activity" genuinely means zero.
    """
    base = (
        labels
        .join(user_features(users), on="user_id", how="inner")
        .join(order_features(orders, cutoff), on="user_id", how="left")
        .join(spend_features(order_items, products, cutoff), on="user_id", how="left")
        .join(event_features(events, cutoff), on="user_id", how="left")
    )
    zero_when_absent = [
        "session_count",
        "avg_events_per_session",
        "cart_events",
        "total_spent",
        "items_bought",
        "distinct_categories",
        "return_rate",
    ]
    return base.fillna(0, subset=zero_when_absent)
