# Runbook

## Daily job (`payments-lakehouse`)
bronze -> silver -> gold -> train -> score, 06:15 Europe/Berlin, one run at a time.

| Symptom | Check | Action |
|---|---|---|
| bronze picked up 0 files | Landing volume for today's folder; `bronze.ingest_log` | Hub delivery issue: contact the payment hub team. Re-running is safe. |
| `well_formed_xml` failures > 0 | `bronze.pacs008 WHERE _corrupt_record IS NOT NULL` | Ask the hub to resend the file. A resend under a new name is de-duplicated on UETR. |
| Spike in `amount_positive` / `uetr_valid` quarantine | `silver.payments_quarantine` grouped by `dq_failed`, `_source_file` | Usually one bad file from an upstream change. Fix upstream, resend, then delete the quarantined rows for that file. |
| `creditor_iban_valid` warnings rising | Compare with AC01 rejects in `gold.reject_reasons` | Expected to match one-to-one. A gap means a validation difference with the clearing: raise with Payment Ops. |
| Many PENDING statuses older than 1 day | `silver.payments WHERE status='PENDING'` by `settlement_date` | pacs.002 delivery is late. Statuses are applied automatically when the files arrive. |
| Train task reports `promoted: false` | MLflow run `evaluation.json` vs. the champion's | The champion keeps scoring. Investigate data drift or label delays before forcing a promotion. |
| CI quality gate fails | Smoke-run metrics in the job log | The change made the model worse than the rules at equal volume. Do not merge. |

## Re-processing
- **Re-run a day:** safe. Bronze skips known files, and the MERGEs are idempotent.
- **Full rebuild of silver:** reset the `ops.watermarks` row for `silver.payments` to 0, truncate silver, run. Bronze is the source of truth.
- **Time travel for audits:** `SELECT * FROM silver.payments VERSION AS OF <n>`. `DESCRIBE HISTORY` shows which run wrote each version.

## Analyst feedback loop
Decisions posted to `/alerts/{uetr}/review` land in `ops.alert_reviews`. Before each monthly retrain, reconcile them with confirmed case outcomes; `FALSE_POSITIVE` decisions are valuable negative labels for exactly the cases the model gets wrong.

## Access requests
Investigators: group `payments_investigators` (clear-text PII in silver). Analysts: `payments_analysts` (gold only). Every grant change goes through the DPO; masks and tags live in `governance/unity_catalog.sql`.
