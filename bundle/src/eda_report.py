# Databricks notebook source
# MAGIC %md
# MAGIC # EDA report — logged to MLflow
# MAGIC
# MAGIC The audits answer yes or no. This task records what the run actually
# MAGIC produced, so a person can look at it later and two runs can be compared:
# MAGIC parameters, metrics and figures, kept in the workspace beside the data
# MAGIC rather than in a CI system that forgets after a retention window.
# MAGIC
# MAGIC The figures come from `rfm/eda.py` — the same functions the pull-request
# MAGIC artefact uses over a fixture. One implementation, two audiences.

# COMMAND ----------

dbutils.widgets.text("catalog", "ctl_training_dev")
dbutils.widgets.text("schema", "m7_dev")
dbutils.widgets.text("table_suffix", "")
dbutils.widgets.text("bundle_root", "")

import json
import sys

import mlflow

BUNDLE_ROOT = dbutils.widgets.get("bundle_root")
if BUNDLE_ROOT and f"{BUNDLE_ROOT}/src" not in sys.path:
    sys.path.append(f"{BUNDLE_ROOT}/src")

from rfm.eda import (figure_monetary_by_quintile, figure_segment_share,  # noqa: E402
                     monetary_by_quintile, segment_share)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
TABLE = f"{CATALOG}.{SCHEMA}.customer_rfm_{SUFFIX}"

USER = spark.sql("SELECT current_user()").collect()[0][0]
EXPERIMENT = f"/Users/{USER}/m7-rfm"

# COMMAND ----------

# MAGIC %md ## The aggregates, computed in Spark and collected small
# MAGIC
# MAGIC Only the aggregates cross to the driver — five segment shares and five
# MAGIC quintile means — never the customer base itself.

# COMMAND ----------

scored = spark.table(TABLE).select(
    "customer_key", "monetary", "m_score", "segment").toPandas()

shares = segment_share(scored)
quintiles = monetary_by_quintile(scored)

# COMMAND ----------

# MAGIC %md ## One MLflow run: parameters, metrics, figures

# COMMAND ----------

mlflow.set_experiment(EXPERIMENT)

with mlflow.start_run(run_name=f"rfm-{SCHEMA}") as run:
    mlflow.log_params({"catalog": CATALOG, "schema": SCHEMA,
                       "table_suffix": SUFFIX, "table": TABLE})

    mlflow.log_metric("rows", len(scored))
    for name, share in shares:
        mlflow.log_metric(f"share_{name.replace(' ', '_').lower()}", share)
    for quintile, mean_value in quintiles:
        mlflow.log_metric("mean_monetary_by_quintile", mean_value,
                          step=quintile)

    mlflow.log_figure(figure_segment_share(shares), "segment_share.png")
    mlflow.log_figure(figure_monetary_by_quintile(quintiles),
                      "monetary_by_quintile.png")

    payload = {"experiment": EXPERIMENT, "run_id": run.info.run_id,
               "rows": len(scored),
               "shares": {name: round(share, 4) for name, share in shares}}

print(json.dumps(payload, indent=2))
dbutils.notebook.exit(json.dumps(payload))
