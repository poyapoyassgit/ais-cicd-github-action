"""Data-quality expectations (DAMA dimensions: completeness, validity,
uniqueness, consistency).

Every function returns a LIST OF VIOLATION STRINGS rather than raising or
returning a boolean, so the same function serves two callers: pytest asserts
the list is empty against a fixture, and the audit notebook fails the task
and reports the list against the real table.
"""

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def completeness_violations(df: DataFrame, columns) -> list:
    """No NULLs where the pipeline promises a value."""
    out = []
    for column in columns:
        n = df.filter(F.col(column).isNull()).count()
        if n:
            out.append(f"completeness: {column} has {n} null value(s)")
    return out


def validity_violations(df: DataFrame, column: str, lo, hi) -> list:
    """Values inside their declared range. NaN is counted separately because
    Spark orders NaN GREATER than any number, so a `> hi` predicate reports
    it as 'above' and a `< lo` predicate never sees it."""
    out = []
    below = df.filter(F.col(column) < F.lit(lo)).count()
    above = df.filter((F.col(column) > F.lit(hi)) & ~F.isnan(F.col(column))).count()
    nan = df.filter(F.isnan(F.col(column))).count()
    if below:
        out.append(f"validity: {column} has {below} value(s) below {lo}")
    if above:
        out.append(f"validity: {column} has {above} value(s) above {hi}")
    if nan:
        out.append(f"validity: {column} has {nan} NaN value(s)")
    return out


def uniqueness_violations(df: DataFrame, key: str) -> list:
    """One row per customer, or the segment counts are meaningless."""
    duplicates = (df.groupBy(key).count().filter(F.col("count") > 1).count())
    if duplicates:
        return [f"uniqueness: {key} repeats on {duplicates} value(s)"]
    return []


def volume_violations(df: DataFrame, min_rows: int) -> list:
    """An empty frame passes every column-level check ever written."""
    n = df.count()
    if n < min_rows:
        return [f"volume: {n} row(s), expected at least {min_rows}"]
    return []


def audit(df: DataFrame) -> list:
    """The full expectation suite for customer_rfm, in one call."""
    return (
        volume_violations(df, 1)
        + completeness_violations(df, ["customer_key", "segment", "monetary"])
        + validity_violations(df, "r_score", 1, 5)
        + validity_violations(df, "f_score", 1, 5)
        + validity_violations(df, "m_score", 1, 5)
        + validity_violations(df, "monetary", 0, float("inf"))
        + uniqueness_violations(df, "customer_key")
    )
