# Databricks notebook source
# MAGIC %md
# MAGIC # Data-quality audit — level 3, against the real table
# MAGIC
# MAGIC The expectations in `rfm/checks.py` were proved against fixtures by
# MAGIC `tests/test_quality.py` on the runner. This task runs the same functions
# MAGIC against the table the build task just wrote, which is the only place the
# MAGIC quality of this morning's data can be established.
# MAGIC
# MAGIC The task FAILS on any violation, so the deployment's own run is red and
# MAGIC nothing downstream proceeds (Write-Audit-Publish).

# COMMAND ----------

dbutils.widgets.text("catalog", "ctl_training_dev")
dbutils.widgets.text("schema", "m7_dev")
dbutils.widgets.text("table_suffix", "")
dbutils.widgets.text("bundle_root", "")

import json
import sys

BUNDLE_ROOT = dbutils.widgets.get("bundle_root")
if BUNDLE_ROOT and f"{BUNDLE_ROOT}/src" not in sys.path:
    sys.path.append(f"{BUNDLE_ROOT}/src")

from rfm.checks import audit  # noqa: E402

TABLE = (f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
         f".customer_rfm_{dbutils.widgets.get('table_suffix').strip()}")

violations = audit(spark.table(TABLE))
payload = {"table": TABLE, "level": "data quality", "violations": violations}
print(json.dumps(payload, indent=2))

if violations:
    raise AssertionError(f"{len(violations)} data-quality violation(s): {violations}")

dbutils.notebook.exit(json.dumps(payload))
