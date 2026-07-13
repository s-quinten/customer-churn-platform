"""Churn label derivation.

TheLook has no "churned" column, so we define churn ourselves, the same way
a real subscription-free retailer would:

    A customer has churned if they placed at least one order, but their
    most recent order is more than CHURN_WINDOW_DAYS before the analysis
    date.

Design decisions:

* Analysis date = max(orders.created_at) IN THE DATA, not today's wall
  clock. The dataset is a static snapshot ending 2024-01-17; measuring
  recency against the real 2026 calendar would mark everyone churned.

* Only customers with >= 1 order can churn. Users who registered but
  never bought are a different business problem (activation, not churn)
  and would poison the label with people who were never customers.

* Cancelled orders don't count as activity. A cancelled order is not a
  purchase; counting it would make a customer look retained when they
  actually walked away at checkout.

* The window (default 180 days) is a business choice, not a statistical
  one. justify_window() prints the inter-purchase gap distribution so
  the choice is grounded in how often these customers actually reorder,
  rather than picked blindly. See the comment on CHURN_WINDOW_DAYS for
  why 180 fits this dataset.
"""
from pyspark.sql import DataFrame
from pyspark.sql import functions as F

# 180 days, chosen from the data: the median gap between a customer's
# consecutive orders is 155 days (justify_window() output, 2026-07-13),
# so 6 months of silence exceeds a typical customer's normal rhythm.
# 90d would label 74% churned (mislabels slow-but-alive customers);
# 180d yields a 60/40 split.
CHURN_WINDOW_DAYS = 180

# Order statuses that count as real purchase activity.
ACTIVE_STATUSES = ("Complete", "Shipped", "Processing", "Returned")


def _valid_orders(orders: DataFrame) -> DataFrame:
    return orders.filter(F.col("status").isin(*ACTIVE_STATUSES))


def analysis_date(orders: DataFrame):
    """The dataset's 'today': the most recent order timestamp."""
    return _valid_orders(orders).agg(F.max("created_at")).first()[0]


def label_churn(
    users: DataFrame,
    orders: DataFrame,
    window_days: int = CHURN_WINDOW_DAYS,
) -> DataFrame:
    """Return one row per purchasing user with a binary `churned` column.

    Output columns: user_id, last_order_at, days_since_last_order, churned
    """
    valid = _valid_orders(orders)
    as_of = analysis_date(orders)

    last_order = (
        valid.groupBy("user_id")
        .agg(F.max("created_at").alias("last_order_at"))
        .withColumn(
            "days_since_last_order",
            F.datediff(F.lit(as_of), F.col("last_order_at")),
        )
    )

    labeled = last_order.withColumn(
        "churned",
        (F.col("days_since_last_order") > window_days).cast("int"),
    )

    # Inner join keeps exactly the users who have >= 1 valid order.
    # Never-purchased users drop out here by design (see module docstring).
    return users.select(F.col("id").alias("user_id")).join(
        labeled, on="user_id", how="inner"
    )


def justify_window(orders: DataFrame) -> DataFrame:
    """Distribution of gaps between consecutive orders per customer.

    If e.g. 80% of repeat purchases happen within 90 days of the previous
    one, then '90 days silent' genuinely separates lapsed customers from
    slow-but-alive ones. Returns a DataFrame of percentiles to inspect.
    """
    from pyspark.sql.window import Window

    valid = _valid_orders(orders)
    w = Window.partitionBy("user_id").orderBy("created_at")
    gaps = (
        valid.withColumn("prev_order_at", F.lag("created_at").over(w))
        .filter(F.col("prev_order_at").isNotNull())
        .withColumn(
            "gap_days", F.datediff("created_at", "prev_order_at")
        )
    )
    return gaps.agg(
        *[
            F.percentile_approx("gap_days", p).alias(f"p{int(p * 100)}")
            for p in (0.25, 0.50, 0.75, 0.80, 0.90, 0.95)
        ]
    )
