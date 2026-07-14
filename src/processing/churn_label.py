"""Churn label derivation.

TheLook has no "churned" column, so we define churn ourselves with a
temporal cutoff to avoid label leakage:

    cutoff = last order date in the data minus CHURN_WINDOW_DAYS

    A user is in scope if they placed at least one valid order on or
    before the cutoff. They are labeled churned (1) if they placed no
    valid order in the window after the cutoff, retained (0) if they did.

Features are computed from data up to the cutoff only, the label comes
from the window after it. Without this split, any recency-style feature
would just restate the label and the model would learn nothing real.

Design decisions:

* The reference point is the max order date IN THE DATA, not today's
  wall clock. The dataset is a static snapshot ending 2024-01-17, so
  measuring against the real calendar would mark everyone churned.

* Only users with >= 1 valid order before the cutoff are labeled.
  Users who registered but never bought are an activation problem, not
  a churn problem, and users whose first purchase falls after the
  cutoff have no history to build features from.

* Cancelled orders never count as activity. A cancelled order is not a
  purchase; counting it would make a customer look retained when they
  actually walked away at checkout.
"""
from datetime import timedelta

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# 180 days, chosen from the data: the median gap between a customer's
# consecutive orders is 155 days (justify_window() output), so 6 months
# of silence exceeds a typical customer's normal rhythm. 90d would call
# most slow-but-alive customers churned.
CHURN_WINDOW_DAYS = 180

# Order statuses that count as real purchase activity.
ACTIVE_STATUSES = ("Complete", "Shipped", "Processing", "Returned")


def valid_orders(orders: DataFrame) -> DataFrame:
    return orders.filter(F.col("status").isin(*ACTIVE_STATUSES))


def cutoff_date(orders: DataFrame, window_days: int = CHURN_WINDOW_DAYS):
    """Cutoff = newest valid order timestamp minus the churn window."""
    max_date = valid_orders(orders).agg(F.max("created_at")).first()[0]
    return max_date - timedelta(days=window_days)


def label_churn(
    orders: DataFrame,
    cutoff,
) -> DataFrame:
    """One row per in-scope user: user_id, churned.

    In scope: >= 1 valid order at or before `cutoff`.
    churned = 1 when the user has no valid order after `cutoff`.
    """
    valid = valid_orders(orders)

    before = (
        valid.filter(F.col("created_at") <= F.lit(cutoff))
        .select("user_id")
        .distinct()
    )
    after = (
        valid.filter(F.col("created_at") > F.lit(cutoff))
        .select("user_id")
        .distinct()
        .withColumn("ordered_again", F.lit(1))
    )

    return (
        before.join(after, on="user_id", how="left")
        .withColumn(
            "churned",
            F.when(F.col("ordered_again").isNull(), 1).otherwise(0),
        )
        .drop("ordered_again")
    )


def justify_window(orders: DataFrame) -> DataFrame:
    """Distribution of gaps between consecutive orders per customer.

    Grounds the window choice in how often customers actually reorder.
    Returns a one-row DataFrame of gap percentiles in days.
    """
    from pyspark.sql.window import Window

    valid = valid_orders(orders)
    w = Window.partitionBy("user_id").orderBy("created_at")
    gaps = (
        valid.withColumn("prev_order_at", F.lag("created_at").over(w))
        .filter(F.col("prev_order_at").isNotNull())
        .withColumn("gap_days", F.datediff("created_at", "prev_order_at"))
    )
    return gaps.agg(
        *[
            F.percentile_approx("gap_days", p).alias(f"p{int(p * 100)}")
            for p in (0.25, 0.50, 0.75, 0.80, 0.90, 0.95)
        ]
    )
