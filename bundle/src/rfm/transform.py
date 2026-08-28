"""Spark transformations. Each takes a DataFrame and returns a DataFrame, so
each can be exercised on a five-row fixture instead of the whole warehouse."""

from pyspark.sql import DataFrame, Window
from pyspark.sql import functions as F

from .logic import SEGMENTS  # noqa: F401  (re-exported for the audits)


def add_net_revenue(lines: DataFrame) -> DataFrame:
    """Revenue lives on the line, not the order (TPC-H)."""
    return lines.withColumn(
        "net_revenue", F.col("l_extendedprice") * (1 - F.col("l_discount")))


def aggregate_rfm(revenue: DataFrame, reference_date) -> DataFrame:
    """One row per customer: recency in days, frequency, monetary value.

    The reference date is passed in — never `current_date()` — so the output
    is a function of the input alone and CI can assert on it.
    """
    agg = revenue.groupBy("o_custkey").agg(
        F.max("o_orderdate").alias("last_order_date"),
        F.countDistinct("o_orderkey").alias("frequency"),
        F.sum("net_revenue").alias("monetary"),
    )
    return agg.withColumn(
        "recency_days",
        F.datediff(F.lit(reference_date), F.col("last_order_date")))


def add_scores(rfm: DataFrame) -> DataFrame:
    """Quintile each measure with NTILE(5), ordered so 5 is always good."""
    return (
        rfm.withColumn("r_score",
                       F.ntile(5).over(Window.orderBy(F.col("recency_days").desc())))
        .withColumn("f_score",
                    F.ntile(5).over(Window.orderBy(F.col("frequency").asc())))
        .withColumn("m_score",
                    F.ntile(5).over(Window.orderBy(F.col("monetary").asc())))
    )


def add_segment(scored: DataFrame) -> DataFrame:
    """The segment rule as column expressions.

    This is the second implementation of logic.segment. Two implementations
    of one rule is a defect waiting to happen, which is exactly why
    tests/test_transform.py pins them against each other over the whole
    5x5x5 score grid (differential testing).
    """
    r, f, m = F.col("r_score"), F.col("f_score"), F.col("m_score")
    return (
        scored.withColumn(
            "rfm_cell",
            F.concat(r.cast("string"), f.cast("string"), m.cast("string")))
        .withColumn(
            "segment",
            F.when((r >= 4) & (f >= 4) & (m >= 4), "Champions")
            .when((r >= 3) & (f >= 3), "Loyal")
            .when((r >= 4) & (f <= 2), "Potential")
            .when((r <= 2) & (f >= 3), "At Risk")
            .otherwise("Hibernating"))
    )
