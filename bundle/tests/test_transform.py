"""LEVEL 2 — transformation tests over the Spark code in rfm/transform.py.

Each test runs one transformation on a fixture of a few rows and checks the
frame that comes back. The point is the transformation, not the warehouse:
these fixtures are small enough to reason about by hand.
"""

import datetime as dt

import pytest

from rfm.logic import segment
from rfm.transform import add_net_revenue, add_scores, add_segment, aggregate_rfm

pytestmark = pytest.mark.spark


def test_add_net_revenue_applies_the_discount_per_line(order_lines):
    out = {(r["o_orderkey"], round(r["net_revenue"], 2))
           for r in add_net_revenue(order_lines).collect()}
    assert (101, 90.0) in out       # 100.00 at 10% off
    assert (101, 200.0) in out      # no discount
    assert (202, 75.0) in out       # 150.00 at 50% off


def test_aggregate_rfm_is_one_row_per_customer(order_lines):
    out = aggregate_rfm(add_net_revenue(order_lines), dt.date(1998, 12, 31))
    assert out.count() == 5
    assert out.select("o_custkey").distinct().count() == 5


def test_aggregate_rfm_counts_orders_not_lines(order_lines):
    """Customer 1 has three lines across two orders: frequency is 2."""
    out = aggregate_rfm(add_net_revenue(order_lines), dt.date(1998, 12, 31))
    row = out.filter("o_custkey = 1").collect()[0]
    assert row["frequency"] == 2
    assert round(row["monetary"], 2) == 530.0        # 90 + 200 + 240


def test_aggregate_rfm_measures_recency_against_the_given_date(order_lines):
    """The reference date is an argument, so the answer never moves."""
    out = aggregate_rfm(add_net_revenue(order_lines), dt.date(1998, 12, 31))
    row = out.filter("o_custkey = 3").collect()[0]
    assert row["last_order_date"] == dt.date(1995, 12, 31)
    assert row["recency_days"] == 1096


def test_add_scores_puts_the_best_customer_in_the_top_quintile(order_lines):
    """Five customers, so NTILE(5) hands out every quintile exactly once."""
    rows = add_scores(aggregate_rfm(add_net_revenue(order_lines),
                                    dt.date(1998, 12, 31))).collect()
    by_customer = {r["o_custkey"]: r for r in rows}
    assert by_customer[1]["r_score"] == 5      # ordered so 5 is always good
    assert by_customer[3]["r_score"] == 1      # the oldest last order
    assert by_customer[5]["m_score"] == 5      # the largest revenue
    assert by_customer[2]["m_score"] == 1
    assert sorted(r["m_score"] for r in rows) == [1, 2, 3, 4, 5]


def test_add_segment_agrees_with_the_pure_rule_on_every_score(spark):
    """Differential testing: the SQL rule and the Python rule are two
    implementations of one specification, so they are pinned against each
    other across the whole 5 x 5 x 5 grid."""
    grid = [(r, f, m) for r in range(1, 6) for f in range(1, 6) for m in range(1, 6)]
    frame = spark.createDataFrame(grid, "r_score int, f_score int, m_score int")

    disagreements = [
        (row["r_score"], row["f_score"], row["m_score"], row["segment"],
         segment(row["r_score"], row["f_score"], row["m_score"]))
        for row in add_segment(frame).collect()
        if row["segment"] != segment(row["r_score"], row["f_score"], row["m_score"])
    ]
    assert disagreements == [], f"{len(disagreements)} of 125 score cells disagree"


def test_add_segment_builds_the_rfm_cell(spark):
    frame = spark.createDataFrame([(5, 4, 3)], "r_score int, f_score int, m_score int")
    assert add_segment(frame).collect()[0]["rfm_cell"] == "543"
