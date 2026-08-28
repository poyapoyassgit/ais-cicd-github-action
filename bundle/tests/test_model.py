"""LEVEL 4 — model validation over the evaluation code in rfm/evaluate.py.

Data quality asks whether the table is well formed. Model validation asks
whether the scoring model still behaves: a perfectly well-formed table whose
segmentation has collapsed onto one segment is a silent failure.
"""

import pytest

from rfm.evaluate import (coverage_violations, monotonicity_violations,
                          segment_distribution, validate)

pytestmark = pytest.mark.spark


def test_distribution_reports_every_segment_including_the_empty_ones(scored_frame):
    dist = segment_distribution(scored_frame)
    assert set(dist) == {"Champions", "Loyal", "Potential", "At Risk", "Hibernating"}
    assert sum(dist.values()) == pytest.approx(1.0)
    assert dist["Champions"] == pytest.approx(0.2)


def test_a_healthy_distribution_passes(scored_frame):
    assert validate(scored_frame) == []


def test_coverage_catches_a_collapsed_segmentation():
    """Every customer in one segment: the table is valid, the model is not."""
    collapsed = {"Champions": 0.0, "Loyal": 1.0, "Potential": 0.0,
                 "At Risk": 0.0, "Hibernating": 0.0}
    reported = coverage_violations(collapsed)
    assert len(reported) == 5          # four empty segments, plus the runaway
    assert "coverage: Champions holds 0.0%, expected at least 1%" in reported
    assert "coverage: Loyal holds 100.0%, expected at most 60%" in reported


def test_coverage_catches_a_runaway_segment():
    runaway = {"Champions": 0.80, "Loyal": 0.05, "Potential": 0.05,
               "At Risk": 0.05, "Hibernating": 0.05}
    assert coverage_violations(runaway) == [
        "coverage: Champions holds 80.0%, expected at most 40%"]


def test_monotonicity_holds_when_value_rises_with_the_quintile(scored_frame):
    assert monotonicity_violations(scored_frame) == []


def test_monotonicity_catches_an_inverted_score(spark, scored_frame):
    """m_score 5 must not be worth less than m_score 4. This is what an
    NTILE ordered the wrong way round looks like from the outside."""
    inverted = spark.createDataFrame(
        [(1, 500.0, 5, 5, 1, "Potential"),
         (2, 400.0, 5, 5, 2, "Loyal"),
         (3, 300.0, 1, 4, 3, "At Risk"),
         (4, 200.0, 2, 1, 4, "Hibernating"),
         (5, 100.0, 5, 5, 5, "Champions")],
        scored_frame.schema)
    reported = monotonicity_violations(inverted)
    assert len(reported) == 4
    assert reported[0] == "monotonicity: mean monetary falls at m_score=2"
