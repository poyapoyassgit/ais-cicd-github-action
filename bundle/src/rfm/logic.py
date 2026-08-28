"""Pure functions: no Spark, no I/O, no clock. Unit-testable in milliseconds.

RFM (Hughes, Strategic Database Marketing, 1994) ranks each customer on
Recency, Frequency and Monetary value. These functions carry the rules that
decide a customer's segment; everything else in the pipeline is plumbing.
"""

SEGMENTS = ["Champions", "Loyal", "Potential", "At Risk", "Hibernating"]


def net_revenue(extended_price: float, discount: float) -> float:
    """TPC-H net revenue for one line item: price after its discount."""
    if extended_price < 0:
        raise ValueError("extended_price must not be negative")
    if not 0.0 <= discount < 1.0:
        raise ValueError("discount must be in [0, 1)")
    return extended_price * (1.0 - discount)


def recency_days(last_order_ordinal: int, reference_ordinal: int) -> int:
    """Days from a customer's last order to the reference date.

    Both arguments are date ordinals, so the function has no clock in it and
    the same inputs always give the same answer.
    """
    days = reference_ordinal - last_order_ordinal
    if days < 0:
        raise ValueError("last order is after the reference date")
    return days


def segment(r_score: int, f_score: int, m_score: int) -> str:
    """The segment rule, as one function with one owner.

    Scores are quintiles, 1 to 5, and 5 is good on every axis.
    """
    for name, value in (("r_score", r_score), ("f_score", f_score),
                        ("m_score", m_score)):
        if not 1 <= value <= 5:
            raise ValueError(f"{name} must be a quintile in 1..5, got {value}")

    if r_score >= 4 and f_score >= 4 and m_score >= 4:
        return "Champions"
    if r_score >= 3 and f_score >= 3:
        return "Loyal"
    if r_score >= 4 and f_score <= 2:
        return "Potential"
    if r_score <= 2 and f_score >= 3:
        return "At Risk"
    return "Hibernating"
