# Databricks notebook source
# MAGIC %md
# MAGIC # Customer RFM from samples.tpch
# MAGIC
# MAGIC Builds a Recency / Frequency / Monetary segmentation of the TPC-H customer
# MAGIC base and writes it to Unity Catalog. Every rule this notebook applies lives
# MAGIC in `src/rfm/`, so the same code the job runs is the code the test suite
# MAGIC covers — the notebook is the imperative shell, `rfm/` is the functional core.

# COMMAND ----------

dbutils.widgets.text("catalog", "ctl_training_dev")
dbutils.widgets.text("schema", "m7_dev")
dbutils.widgets.text("table_suffix", "")
dbutils.widgets.text("bundle_root", "")

import json
import sys

from pyspark.sql import functions as F

BUNDLE_ROOT = dbutils.widgets.get("bundle_root")
if BUNDLE_ROOT and f"{BUNDLE_ROOT}/src" not in sys.path:
    sys.path.append(f"{BUNDLE_ROOT}/src")

from rfm.transform import (add_net_revenue, add_scores, add_segment,  # noqa: E402
                           aggregate_rfm)

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SUFFIX = dbutils.widgets.get("table_suffix").strip()
if not SUFFIX:
    raise ValueError("table_suffix is required: two people writing one table is not a result")

RFM_TABLE = f"{CATALOG}.{SCHEMA}.customer_rfm_{SUFFIX}"
print(f"writing {RFM_TABLE}")

# COMMAND ----------

# MAGIC %md ## 1 · Read, and compute the three measures
# MAGIC
# MAGIC The reference date is `MAX(o_orderdate)`, not today. TPC-H order dates run
# MAGIC from 1992 to 1998; measured against `current_date()` every customer scores
# MAGIC about eleven thousand days, the quintiles collapse, and the output stops
# MAGIC being reproducible — which would also stop CI asserting anything about it.

# COMMAND ----------

orders = spark.table("samples.tpch.orders")
lineitem = spark.table("samples.tpch.lineitem")

lines = orders.join(lineitem, orders.o_orderkey == lineitem.l_orderkey).select(
    "o_custkey", "o_orderkey", "o_orderdate", "l_extendedprice", "l_discount")

REFERENCE_DATE = orders.agg(F.max("o_orderdate")).collect()[0][0]
print("reference date:", REFERENCE_DATE)

rfm = aggregate_rfm(add_net_revenue(lines), REFERENCE_DATE)

# COMMAND ----------

# MAGIC %md ## 2 · Score, segment, and join the customer dimension

# COMMAND ----------

scored = add_segment(add_scores(rfm)).withColumnRenamed("o_custkey", "customer_key")

customer = spark.table("samples.tpch.customer")
nation = spark.table("samples.tpch.nation")
region = spark.table("samples.tpch.region")

customer_dim = (
    customer.join(nation, customer.c_nationkey == nation.n_nationkey)
    .join(region, nation.n_regionkey == region.r_regionkey)
    .select(
        F.col("c_custkey").alias("customer_key"),
        F.col("c_name").alias("customer_name"),
        F.col("c_mktsegment").alias("market_segment"),
        F.col("n_name").alias("nation"),
        F.col("r_name").alias("region"),
    )
)

result = (scored.join(customer_dim, "customer_key", "inner")
          .withColumn("reference_date", F.lit(REFERENCE_DATE)))

# COMMAND ----------

# MAGIC %md ## 3 · Write, and report through dbutils.notebook.exit
# MAGIC
# MAGIC `print()` does not reach the Jobs API; `dbutils.notebook.exit` is the only
# MAGIC channel to `notebook_output.result`, which is what CI reads.

# COMMAND ----------

# Idempotent, and it needs the CREATE SCHEMA privilege on the catalog. Where
# the credential does not have it, an administrator creates m7_dev and
# m7_prod once and this line is a no-op.
spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
result.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(RFM_TABLE)

payload = {"rows": result.count(), "reference_date": str(REFERENCE_DATE),
           "table": RFM_TABLE}
print(json.dumps(payload, indent=2))
dbutils.notebook.exit(json.dumps(payload))
