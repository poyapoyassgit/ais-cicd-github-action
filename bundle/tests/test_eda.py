"""LEVEL 1 (continued) — the numbers behind the figures.

A picture cannot be asserted on, but the aggregation that produces it can.
These tests run without Spark and without a workspace, which is exactly why
the EDA report can be produced on a GitHub runner.
"""

import pytest

from rfm.eda import monetary_by_quintile, segment_share, write_report
from rfm.fixture import scored_frame


def test_segment_share_reports_every_segment_and_sums_to_one():
    shares = segment_share(scored_frame())
    assert [name for name, _ in shares] == [
        "Champions", "Loyal", "Potential", "At Risk", "Hibernating"]
    assert sum(share for _, share in shares) == pytest.approx(1.0)


def test_segment_share_of_an_empty_frame_is_all_zero():
    empty = scored_frame().iloc[0:0]
    assert segment_share(empty) == [
        (name, 0.0) for name in
        ["Champions", "Loyal", "Potential", "At Risk", "Hibernating"]]


def test_monetary_rises_with_the_quintile():
    points = monetary_by_quintile(scored_frame())
    assert [q for q, _ in points] == [1, 2, 3, 4, 5]
    values = [v for _, v in points]
    assert values == sorted(values)


def test_write_report_writes_two_figures_and_a_summary(tmp_path):
    written = write_report(scored_frame(), tmp_path / "eda")
    assert [p.name for p in written] == [
        "segment_share.png", "monetary_by_quintile.png", "summary.md"]
    assert all(p.exists() and p.stat().st_size > 0 for p in written)
    assert "RFM segment" not in (tmp_path / "eda" / "summary.md").read_text()
