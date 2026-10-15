-- Unity Catalog governance for the payments lakehouse (run once per catalog by a metastore/catalog admin).
-- Groups: payments_engineers (build/run), payments_investigators (fraud/AML case work),
--         payments_analysts (ops reporting), payments_lakehouse-sp (job service principal).
-- Principle: raw identifiers stay in bronze/silver behind masks; gold is PII-free by construction.

USE CATALOG ${catalog};

-- 1. Access by layer --------------------------------------------------------------------------
GRANT USE CATALOG ON CATALOG ${catalog} TO `payments_analysts`, `payments_investigators`, `payments_engineers`;
GRANT USE SCHEMA, SELECT ON SCHEMA gold TO `payments_analysts`, `payments_investigators`;
GRANT USE SCHEMA, SELECT ON SCHEMA silver TO `payments_investigators`;
GRANT ALL PRIVILEGES ON SCHEMA bronze TO `payments_engineers`;
GRANT ALL PRIVILEGES ON SCHEMA silver TO `payments_engineers`;
GRANT ALL PRIVILEGES ON SCHEMA gold TO `payments_engineers`;
GRANT ALL PRIVILEGES ON SCHEMA ops TO `payments_engineers`;
GRANT MODIFY ON TABLE ops.alert_reviews TO `payments_investigators`;

-- 2. Column masks on silver PII (investigators see clear text, everyone else a masked value) ---
CREATE OR REPLACE FUNCTION silver.mask_iban(iban STRING)
  RETURNS STRING
  RETURN CASE WHEN is_account_group_member('payments_investigators') THEN iban
              ELSE concat(substr(iban, 1, 4), repeat('*', greatest(length(iban) - 8, 0)), right(iban, 4)) END;

CREATE OR REPLACE FUNCTION silver.mask_name(name STRING)
  RETURNS STRING
  RETURN CASE WHEN is_account_group_member('payments_investigators') THEN name ELSE '***' END;

ALTER TABLE silver.payments ALTER COLUMN debtor_iban   SET MASK silver.mask_iban;
ALTER TABLE silver.payments ALTER COLUMN creditor_iban SET MASK silver.mask_iban;
ALTER TABLE silver.payments ALTER COLUMN debtor_name   SET MASK silver.mask_name;
ALTER TABLE silver.payments ALTER COLUMN creditor_name SET MASK silver.mask_name;
ALTER TABLE silver.payments ALTER COLUMN remittance_info SET MASK silver.mask_name;

-- 3. Classification tags (drive discovery, lineage review and DPIA evidence) ------------------
ALTER TABLE silver.payments ALTER COLUMN debtor_iban   SET TAGS ('pii' = 'account_identifier');
ALTER TABLE silver.payments ALTER COLUMN creditor_iban SET TAGS ('pii' = 'account_identifier');
ALTER TABLE silver.payments ALTER COLUMN debtor_name   SET TAGS ('pii' = 'person_name');
ALTER TABLE silver.payments ALTER COLUMN creditor_name SET TAGS ('pii' = 'person_name');
ALTER TABLE silver.payments ALTER COLUMN debtor_ref    SET TAGS ('pii' = 'pseudonym_hmac');
ALTER TABLE silver.payments SET TAGS ('retention' = 'statutory_gwg_hgb', 'data_owner' = 'payments_operations');
ALTER TABLE gold.payment_features SET TAGS ('retention' = 'derived_400d');

-- 4. Row filter: analysts only see payments outside an active legal hold ---------------------
CREATE TABLE IF NOT EXISTS ops.legal_hold (uetr STRING, reason STRING, placed_at TIMESTAMP);
CREATE OR REPLACE FUNCTION gold.not_on_hold(uetr STRING)
  RETURNS BOOLEAN
  RETURN is_account_group_member('payments_investigators')
      OR NOT EXISTS (SELECT 1 FROM ops.legal_hold h WHERE h.uetr = uetr);
ALTER TABLE gold.payments_analytics SET ROW FILTER gold.not_on_hold ON (uetr);
