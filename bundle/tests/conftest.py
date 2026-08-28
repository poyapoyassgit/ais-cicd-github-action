"""One local Spark session for the whole suite: starting it is the expensive
part, so it is session-scoped.

Levels 2, 3 and 4 need a JVM. On a machine without one they are SKIPPED with
a single explanatory line rather than raising JAVA_GATEWAY_EXITED once per
test — a participant should be told what to install, not shown eighteen
tracebacks. CI sets M7_REQUIRE_SPARK=1, which turns the skip back into a
failure: a runner silently skipping the Spark levels would be a gate that
proves less than it appears to.
"""

import datetime as dt
import os
import shutil
import subprocess

import pytest
from pyspark.sql import SparkSession


def _java_version():
    """The major version of the java on PATH, or None if there is none."""
    java = shutil.which("java")
    if not java:
        return None
    try:
        out = subprocess.run([java, "-version"], capture_output=True, text=True,
                             timeout=30).stderr
    except (OSError, subprocess.SubprocessError):
        return None
    for token in out.split():
        if token.startswith('"'):
            digits = token.strip('"').split(".")[0]
            return int(digits) if digits.isdigit() else None
    return None


def pytest_collection_modifyitems(config, items):
    version = _java_version()
    if version is not None and version >= 17:
        return
    if os.environ.get("M7_REQUIRE_SPARK") == "1":
        return                      # CI: let the failure be a failure
    reason = (
        "no Java 17+ on this machine, so the Spark levels cannot run "
        "(sudo apt-get install -y openjdk-17-jre-headless). Level 1 still "
        "runs, and CI runs the rest."
        if version is None else
        f"Java {version} is too old for pyspark 3.5.x; 17 or later is needed."
    )
    skip = pytest.mark.skip(reason=reason)
    for item in items:
        if "spark" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def spark():
    session = (SparkSession.builder
               .master("local[1]")
               .appName("m7demo-tests")
               .config("spark.ui.enabled", "false")
               .config("spark.sql.shuffle.partitions", "1")
               .getOrCreate())
    yield session
    session.stop()


@pytest.fixture
def scored_frame(spark):
    """A deterministic scored frame: five customers, one per segment, with
    monetary rising with m_score."""
    rows = [
        # customer_key, monetary, r, f, m, segment
        (1, 100.0, 5, 5, 1, "Potential"),
        (2, 200.0, 5, 5, 2, "Loyal"),
        (3, 300.0, 1, 4, 3, "At Risk"),
        (4, 400.0, 2, 1, 4, "Hibernating"),
        (5, 500.0, 5, 5, 5, "Champions"),
    ]
    return spark.createDataFrame(
        rows,
        "customer_key int, monetary double, r_score int, f_score int, "
        "m_score int, segment string")


@pytest.fixture
def order_lines(spark):
    """Five customers, nine order lines, dates inside the TPC-H range.

    Five is the smallest fixture on which NTILE(5) produces every quintile,
    so the score assertions below mean what they say.
    """
    d = dt.date
    rows = [
        # custkey, orderkey, orderdate, extendedprice, discount
        (1, 101, d(1998, 1, 5), 100.0, 0.10),     # -> 90
        (1, 101, d(1998, 1, 5), 200.0, 0.00),     # -> 200
        (1, 102, d(1998, 6, 1), 300.0, 0.20),     # -> 240   c1: 2 orders, 530
        (2, 201, d(1997, 3, 3), 50.0, 0.00),      # -> 50
        (2, 202, d(1997, 9, 9), 150.0, 0.50),     # -> 75    c2: 2 orders, 125
        (3, 301, d(1995, 12, 31), 400.0, 0.25),   # -> 300   c3: 1 order,  300
        (4, 401, d(1998, 3, 15), 250.0, 0.00),    # -> 250
        (4, 402, d(1998, 3, 15), 100.0, 0.00),    # -> 100
        (4, 403, d(1997, 11, 2), 50.0, 0.00),     # -> 50    c4: 3 orders, 400
        (5, 501, d(1996, 7, 4), 600.0, 0.10),     # -> 540   c5: 1 order,  540
    ]
    return spark.createDataFrame(
        rows,
        "o_custkey int, o_orderkey int, o_orderdate date, "
        "l_extendedprice double, l_discount double")
