"""Exploratory figures over a scored RFM frame.

The aggregation is separated from the drawing, so the numbers behind every
figure are unit-testable (tests/test_eda.py) even though the picture itself
is not. Everything here takes and returns pandas, never Spark: the same
functions run on the driver inside Databricks and on a GitHub runner over a
fixture.
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")          # no display on a runner or a driver
import matplotlib.pyplot as plt  # noqa: E402

from .logic import SEGMENTS  # noqa: E402

ORANGE = "#FC7800"
INK = "#222224"


def segment_share(scored) -> "list[tuple[str, float]]":
    """Share of customers per segment, every segment present, in the fixed
    order of SEGMENTS so two runs are comparable."""
    total = len(scored)
    counts = scored["segment"].value_counts().to_dict() if total else {}
    return [(name, (counts.get(name, 0) / total) if total else 0.0)
            for name in SEGMENTS]


def monetary_by_quintile(scored) -> "list[tuple[int, float]]":
    """Mean monetary value per m_score, ascending. The visual form of the
    monotonicity expectation that rfm/evaluate.py asserts."""
    grouped = scored.groupby("m_score")["monetary"].mean().sort_index()
    return [(int(k), float(v)) for k, v in grouped.items()]


def figure_segment_share(shares):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.bar([name for name, _ in shares], [share for _, share in shares],
           color=ORANGE)
    ax.set_ylabel("share of customers")
    ax.set_title("RFM segment distribution")
    ax.set_ylim(0, 1)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def figure_monetary_by_quintile(points):
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=150)
    ax.plot([q for q, _ in points], [v for _, v in points],
            marker="o", color=INK)
    ax.set_xlabel("monetary quintile (m_score)")
    ax.set_ylabel("mean monetary value")
    ax.set_title("Monetary value by quintile")
    ax.set_xticks([q for q, _ in points])
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    return fig


def write_report(scored, out_dir) -> "list[Path]":
    """Write both figures and a short summary. Returns the paths written."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    shares = segment_share(scored)
    quintiles = monetary_by_quintile(scored)

    written = []
    for fig, name in ((figure_segment_share(shares), "segment_share.png"),
                      (figure_monetary_by_quintile(quintiles),
                       "monetary_by_quintile.png")):
        path = out / name
        fig.savefig(path)
        plt.close(fig)
        written.append(path)

    summary = out / "summary.md"
    lines = ["# EDA summary", "", f"Rows: {len(scored)}", "",
             "| segment | share |", "|---|---|"]
    lines += [f"| {name} | {share:.1%} |" for name, share in shares]
    lines += ["", "| m_score | mean monetary |", "|---|---|"]
    lines += [f"| {q} | {v:,.2f} |" for q, v in quintiles]
    summary.write_text("\n".join(lines) + "\n")
    written.append(summary)
    return written
