# Engagement brief (fictional customer)

**Customer:** a mid-sized German retail bank ("PLH Bank") processing roughly 4-5k outgoing SEPA credit transfers per day through its payment hub.
**Sponsor:** Head of Payment Operations. **Stakeholders:** Fraud & AML investigations, Data Protection Officer, Platform/IT security, Internal Audit.

## Problem as the customer described it

1. Fraud monitoring runs on four threshold rules. Investigators say most alerts are noise, and APP scams and money-mule payments still get through.
2. Payment data sits in hub logs and XML archives. Operations cannot answer simple questions ("what was our reject rate to NL last week?") without a ticket to IT.
3. Any new platform must satisfy the DPO (GDPR, purpose limitation) and audit (traceability of every number and every model decision).

## What we agreed to deliver

| # | Deliverable | Acceptance criterion |
|---|---|---|
| 1 | Incremental ISO 20022 ingestion into Delta (bronze/silver/gold) | Re-running never duplicates; malformed and re-sent files are handled without manual work |
| 2 | Data-quality rules owned by Payment Ops | Every quarantined row carries the failed rule; failure rates visible per rule |
| 3 | Pseudonymised analytics layer | No raw IBAN, name or remittance text in gold, features, alerts or copilot |
| 4 | Fraud/mule model | Beats the current rules at the same alert volume on a time-based holdout; investigators get reason codes |
| 5 | Ops copilot | Read-only, cannot touch PII or non-gold tables, every answer shows its SQL |
| 6 | Deployable, repeatable | Asset Bundle with dev/prod targets; CI blocks merges that degrade the model |

## Success measures agreed with the sponsor

- Precision and recall **at the investigators' capacity** (1% of daily payments), not accuracy.
- Recall per fraud typology, so a gain on one pattern cannot hide a loss on another.
- Time from "new file lands" to "alert available" under one scheduled run.

## Out of scope for this phase

Real-time (sub-second) scoring at payment initiation, inbound payments, sanctions screening, and production case-management integration. The design keeps these open: features are computed with event-time windows that a streaming job can reuse, and alerts carry stable UETR keys for a case system.

## Handover

The bank's data team runs the job and owns the rules registry (`expectations.py`); investigators own `ops.alert_reviews`, which becomes the labelled training feedback; the runbook covers the operational procedures.
