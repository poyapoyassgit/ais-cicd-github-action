"""LEVEL 1 — unit tests over the pure functions in rfm/logic.py.

No Spark, no data, no workspace. Milliseconds, and they run on every push.
"""

import pytest

from rfm.logic import net_revenue, recency_days, segment


# --- net_revenue: Arrange, Act, Assert -------------------------------------
def test_net_revenue_applies_the_discount():
    assert net_revenue(100.0, 0.25) == 75.0


def test_net_revenue_without_a_discount_is_the_price():
    assert net_revenue(100.0, 0.0) == 100.0


@pytest.mark.parametrize("price,discount", [(-1.0, 0.0), (100.0, 1.0), (100.0, -0.1)])
def test_net_revenue_rejects_impossible_inputs(price, discount):
    """Negative testing (ISTQB): the invalid case must fail loudly, not
    silently produce a number."""
    with pytest.raises(ValueError):
        net_revenue(price, discount)


# --- recency_days ----------------------------------------------------------
def test_recency_days_counts_backwards_from_the_reference_date():
    assert recency_days(100, 130) == 30


def test_recency_days_rejects_an_order_after_the_reference_date():
    with pytest.raises(ValueError):
        recency_days(200, 100)


# --- segment: boundary value analysis on the quintile thresholds -----------
@pytest.mark.parametrize("r,f,m,expected", [
    (5, 5, 5, "Champions"),
    (4, 4, 4, "Champions"),      # on the boundary, inside
    (4, 4, 3, "Loyal"),          # one step below the Champions boundary
    (3, 3, 1, "Loyal"),
    (5, 1, 5, "Potential"),
    (4, 2, 1, "Potential"),
    (1, 5, 5, "At Risk"),
    (2, 3, 1, "At Risk"),
    (1, 1, 1, "Hibernating"),
    (3, 2, 5, "Hibernating"),    # matches no rule, so it falls through
])
def test_segment_rule(r, f, m, expected):
    assert segment(r, f, m) == expected


@pytest.mark.parametrize("r,f,m", [(0, 3, 3), (6, 3, 3), (3, 3, 9)])
def test_segment_rejects_scores_outside_the_quintiles(r, f, m):
    with pytest.raises(ValueError):
        segment(r, f, m)
