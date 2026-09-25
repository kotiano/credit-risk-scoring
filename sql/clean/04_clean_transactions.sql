-- =====================================================================================
-- clean.transactions / clean.rejected_transactions
-- Source : staging.stg_transactions
-- Grain  : one row per wallet transaction
-- Rules  : quarantine rows with missing amounts, dates after the extract snapshot, or
--          unknown borrowers. Amount outliers are retained with `is_amount_outlier`
--          (features use robust aggregates; see docs/feature-engineering.md).
-- =====================================================================================
CREATE OR REPLACE TABLE clean.rejected_transactions AS
SELECT
    *,
    concat_ws(', ',
        CASE WHEN NOT is_amount_present THEN 'missing_amount' END,
        CASE WHEN NOT is_date_valid THEN 'future_or_invalid_date' END,
        CASE WHEN NOT has_parent_borrower THEN 'unknown_borrower' END
    )                                                        AS rejection_reason
FROM staging.stg_transactions
WHERE NOT (is_amount_present AND is_date_valid AND has_parent_borrower);

CREATE OR REPLACE TABLE clean.transactions AS
SELECT
    transaction_id,
    borrower_id,
    transaction_date,
    transaction_type,
    direction,
    amount,
    abs(amount)                                              AS abs_amount,
    balance,
    channel,
    is_amount_outlier
FROM staging.stg_transactions
WHERE is_amount_present AND is_date_valid AND has_parent_borrower;
