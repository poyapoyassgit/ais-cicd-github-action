"""python -m rfm.eda-style entry point: write the EDA report for a frame the
runner can build itself. Used by the artefact workflow (E6)."""

import argparse

from .eda import write_report
from .fixture import scored_frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Write the EDA report")
    parser.add_argument("--out", default="reports/eda")
    args = parser.parse_args()
    for path in write_report(scored_frame(), args.out):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
