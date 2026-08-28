"""A deterministic scored frame, in pandas, for anything that must run
without Spark and without a workspace — the EDA report a pull request
carries, for instance."""

import pandas as pd

from .logic import segment

_ROWS = 200


def scored_frame(rows: int = _ROWS):
    """A reproducible customer base: quintiles cycle, monetary rises with
    m_score, and the segment column comes from the same rule the pipeline
    uses, so the picture is a picture of the RULE, not of invented data."""
    records = []
    for i in range(rows):
        r = (i % 5) + 1
        f = ((i // 5) % 5) + 1
        m = ((i // 25) % 5) + 1
        records.append({
            "customer_key": i + 1,
            "monetary": 100.0 * m + (i % 7),
            "r_score": r, "f_score": f, "m_score": m,
            "segment": segment(r, f, m),
        })
    return pd.DataFrame.from_records(records)
