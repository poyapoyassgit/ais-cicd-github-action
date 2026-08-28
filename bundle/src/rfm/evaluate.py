"""Model validation: offline evaluation of the scoring model itself.

A data-quality check asks whether the table is well formed. Model validation
asks whether the model still behaves — the segmentation could be perfectly
well formed and still have collapsed onto one segment overnight.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F

from .logic import SEGMENTS

# Share of the customer base each segment is expected to hold. Wide bounds:
# the point is to catch collapse and runaway, not to freeze the distribution.
EXPECTED_SHARE = {
    "Champions": (0.01, 0.40),
    "Loyal": (0.01, 0.60),
    "Potential": (0.01, 0.40),
    "At Risk": (0.01, 0.40),
    "Hibernating": (0.01, 0.60),
}


def segment_distribution(scored: DataFrame) -> dict:
    """Share of customers per segment; every segment always present, so a
    missing segment is a zero rather than a missing key."""
    total = scored.count()
    if total == 0:
        return {name: 0.0 for name in SEGMENTS}
    counts = {row["segment"]: row["n"]
              for row in scored.groupBy("segment").agg(F.count("*").alias("n")).collect()}
    return {name: counts.get(name, 0) / total for name in SEGMENTS}


def coverage_violations(distribution: dict, bounds=None) -> list:
    """No segment may vanish or swallow the base."""
    bounds = bounds or EXPECTED_SHARE
    out = []
    for name in SEGMENTS:
        share = distribution.get(name, 0.0)
        lo, hi = bounds[name]
        if share < lo:
            out.append(f"coverage: {name} holds {share:.1%}, expected at least {lo:.0%}")
        elif share > hi:
            out.append(f"coverage: {name} holds {share:.1%}, expected at most {hi:.0%}")
    return out


def monotonicity_violations(scored: DataFrame) -> list:
    """Mean monetary value must rise with the monetary quintile. If it does
    not, the scoring is inverted or the quintiles were computed on the wrong
    column — a defect no schema check can see."""
    rows = (scored.groupBy("m_score")
            .agg(F.avg("monetary").alias("avg_monetary"))
            .orderBy("m_score").collect())
    out = []
    previous = None
    for row in rows:
        if previous is not None and row["avg_monetary"] < previous:
            out.append(
                f"monotonicity: mean monetary falls at m_score={row['m_score']}")
        previous = row["avg_monetary"]
    return out


def validate(scored: DataFrame) -> list:
    """The full model-validation suite, in one call."""
    return (coverage_violations(segment_distribution(scored))
            + monotonicity_violations(scored))
