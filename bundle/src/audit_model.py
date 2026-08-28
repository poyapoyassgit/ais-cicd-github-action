# Databricks notebook source
# MAGIC %md
# MAGIC # Model validation — level 4, against the real table
# MAGIC
# MAGIC `rfm/evaluate.py` asks whether the scoring model still behaves: every
# MAGIC segment populated, no segment swallowing the base, and monetary value
# MAGIC rising with the monetary quintile. A table can pass every data-quality
# MAGIC expectation and still fail here, which is the distinction between the
# MAGIC two levels.

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

from rfm.evaluate import segment_distribution, validate  # noqa: E402

TABLE = (f"{dbutils.widgets.get('catalog')}.{dbutils.widgets.get('schema')}"
         f".customer_rfm_{dbutils.widgets.get('table_suffix').strip()}")

scored = spark.table(TABLE)
violations = validate(scored)
payload = {
    "table": TABLE,
    "level": "model validation",
    "distribution": {k: round(v, 4) for k, v in segment_distribution(scored).items()},
    "violations": violations,
}
print(json.dumps(payload, indent=2))

if violations:
    raise AssertionError(f"{len(violations)} model-validation violation(s): {violations}")

dbutils.notebook.exit(json.dumps(payload))
