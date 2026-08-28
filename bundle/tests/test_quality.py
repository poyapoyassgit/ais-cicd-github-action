"""LEVEL 3 — data-quality tests over the expectations in rfm/checks.py.

The expectations are what the audit task runs against the real table after
every deployment. These tests prove the expectations themselves work: each
one is shown passing on a clean fixture and catching an injected fault.
"""

import pytest

from rfm.checks import (audit, completeness_violations, uniqueness_violations,
                        validity_violations, volume_violations)

pytestmark = pytest.mark.spark


def test_a_clean_frame_raises_nothing(scored_frame):
    assert audit(scored_frame) == []


def test_completeness_catches_an_injected_null(spark, scored_frame):
    dirty = scored_frame.union(
        spark.createDataFrame([(6, 600.0, 3, 3, 3, None)],
                              scored_frame.schema))
    assert completeness_violations(dirty, ["segment"]) == [
        "completeness: segment has 1 null value(s)"]


def test_validity_catches_a_score_outside_the_quintiles(spark, scored_frame):
    dirty = scored_frame.union(
        spark.createDataFrame([(7, 700.0, 9, 3, 3, "Loyal")], scored_frame.schema))
    assert validity_violations(dirty, "r_score", 1, 5) == [
        "validity: r_score has 1 value(s) above 5"]


def test_validity_counts_nan_separately(spark, scored_frame):
    """Spark orders NaN greater than any number, so without its own predicate
    a NaN is reported as 'above the upper bound' — or missed entirely."""
    dirty = scored_frame.union(
        spark.createDataFrame([(8, float("nan"), 3, 3, 3, "Loyal")],
                              scored_frame.schema))
    assert validity_violations(dirty, "monetary", 0, float("inf")) == [
        "validity: monetary has 1 NaN value(s)"]


def test_uniqueness_catches_a_duplicated_customer(spark, scored_frame):
    dirty = scored_frame.union(scored_frame.limit(1))
    assert uniqueness_violations(dirty, "customer_key") == [
        "uniqueness: customer_key repeats on 1 value(s)"]


def test_volume_catches_the_empty_frame(spark, scored_frame):
    """An empty frame passes every column-level expectation, which is why a
    row-count expectation has to exist."""
    empty = spark.createDataFrame([], scored_frame.schema)
    assert completeness_violations(empty, ["segment"]) == []
    assert volume_violations(empty, 1) == ["volume: 0 row(s), expected at least 1"]


def test_the_full_audit_reports_every_fault_at_once(spark, scored_frame):
    dirty = scored_frame.union(
        spark.createDataFrame([(1, None, 9, 3, 3, "Loyal")], scored_frame.schema))
    reported = audit(dirty)
    assert len(reported) == 3
    assert any(v.startswith("completeness: monetary") for v in reported)
    assert any(v.startswith("validity: r_score") for v in reported)
    assert any(v.startswith("uniqueness") for v in reported)
