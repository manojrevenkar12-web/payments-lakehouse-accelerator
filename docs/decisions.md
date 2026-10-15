# Architecture decisions

### 1. Medallion layers with a strict PII boundary between silver and gold
Raw identifiers (IBANs, names, remittance text) are needed for investigations and for clearing reconciliation, so they stay in bronze/silver behind Unity Catalog column masks. Gold is PII-free *by construction*: parties appear only as keyed HMAC-SHA256 tokens. Joins still work because tokens are deterministic, but nobody can reverse or recompute them without the secret-scope key. ML features, alerts and the copilot read only gold, which keeps the DPIA scope small.

### 2. Explicit XML schemas, never inference
Schema inference on XML is slow, can differ between batches, and silently types amounts as `DOUBLE`. Money is `DECIMAL(18,2)` from the first read. Malformed files land as `_corrupt_record` rows instead of failing the run.

### 3. Exactly-once ingestion and idempotent writes
Files are picked up once: Auto Loader's checkpoint on Databricks, a file manifest (`bronze.ingest_log`) locally. Silver is a MERGE on UETR (insert-only), so re-sent files and job retries are absorbed. In the reference run, 1,499 duplicates from three re-sent files were absorbed with no manual action.

### 4. Status is recomputed, not appended
pacs.002 reports can arrive late or out of order. Each run rebuilds the latest status per UETR from all status events and MERGEs only rows whose status changed. Order of arrival does not matter, and a re-run updates nothing.

### 5. Data-quality rules as data, two execution engines
Rules are a list of SQL boolean expressions with an action (`quarantine` or `warn`). The same registry drives the job's split into clean and quarantined rows and the `@dlt.expect_all*` decorators in the Lakeflow version. Payment Ops can review the rules without reading Spark code. A bad creditor IBAN is a *warning*, not a quarantine: the payment was really attempted and is relevant for fraud, and the clearing will reject it with AC01, which the run then confirms.

### 6. Point-in-time features only
Every behavioural feature uses event-time windows that end at the payment itself (`rangeBetween(..., 0)`), and the amount profile uses strictly earlier history (`-1`). Train and test are split by date. Unit tests on a hand-built example check that no window looks ahead.

### 7. Evaluate at the analysts' capacity, against the incumbent
The fraud team can review about 1% of daily payments, so the headline metrics are precision and recall at that budget. The model is also compared with the existing rules *at the rules' own alert volume*, and with a hybrid where the rules are kept and the model fills the remaining budget. In the reference run the hybrid was clearly worse than the model alone, which turned "should we keep the old rules?" into a measured answer.

### 8. Champion/challenger promotion
Every training run is logged and registered in MLflow. The `champion` alias moves only when the new version's PR-AUC is not worse than the current champion's (0.005 tolerance). CI also fails if the model stops beating the rules at equal volume. Scoring always loads `@champion`, so a bad retrain never reaches investigators.

### 9. Copilot: the LLM is untrusted
The model only proposes SQL. `sqlglot` parses it and allows exactly one read-only SELECT over allowlisted gold tables, with no PII column names, no denylisted functions (`reflect`, `read_files`, ...) and a forced LIMIT. Logical table names are rewritten to physical ones after validation, and the executed SQL is returned with every answer for audit. Quality is measured by execution match on a fixed question set, and safety by a must-refuse set.

### 10. Retention
Payment records are subject to statutory retention (AML record keeping under GwG §8, bookkeeping under HGB/AO), so erasure requests generally cannot delete them. Instead, access is minimised (masks, gold-only analytics) and derived data has its own, shorter retention (`paylake retention`, default 400 days), followed by VACUUM.

### 11. Spark choices
- Broadcast join for the BIC directory (small reference table).
- AQE with partition coalescing; shuffle partitions sized to the cluster.
- Silver and features partitioned by `settlement_date`, the dominant filter for ops queries and retention deletes.
- `collect_set` over a 24h window gives exact distinct-payer counts. At higher volumes, switch to `approx_count_distinct` per hourly bucket, or to pre-aggregated creditor-hour tables.
- At bank scale, compute features incrementally over a lookback window, not by rebuilding gold from all of silver. The window definitions already bound the lookback (30 days).
