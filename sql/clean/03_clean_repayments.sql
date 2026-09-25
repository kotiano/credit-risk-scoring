-- =====================================================================================
-- clean.repayments / clean.rejected_repayments
-- Source : staging.stg_repayments
-- Grain  : one row per payment event
-- Rules  : quarantine payments that cannot be attributed to a valid loan (orphans,
--          payments on quarantined loans), non-positive amounts, payments dated before
--          disbursement, and borrower mismatches.
-- =====================================================================================
CREATE OR REPLACE TABLE clean.rejected_repayments AS
SELECT
    r.*,
    concat_ws(', ',
        CASE WHEN NOT r.has_parent_loan THEN 'orphan_loan_id' END,
        CASE WHEN r.has_parent_loan AND l.loan_id IS NULL THEN 'parent_loan_quarantined' END,
        CASE WHEN NOT r.is_amount_valid THEN 'non_positive_amount' END,
        CASE WHEN r.has_parent_loan AND NOT r.is_date_valid THEN 'before_disbursement' END,
        CASE WHEN r.has_parent_loan AND NOT r.is_borrower_consistent
             THEN 'borrower_mismatch' END
    )                                                        AS rejection_reason
FROM staging.stg_repayments AS r
LEFT JOIN clean.loans AS l
    ON r.loan_id = l.loan_id
WHERE l.loan_id IS NULL
   OR NOT (r.is_amount_valid AND r.is_date_valid AND r.is_borrower_consistent);

CREATE OR REPLACE TABLE clean.repayments AS
SELECT
    r.repayment_id,
    r.loan_id,
    r.borrower_id,
    r.repayment_date,
    r.amount,
    r.payment_channel,
    r.transaction_reference
FROM staging.stg_repayments AS r
WHERE r.repayment_id NOT IN (SELECT repayment_id FROM clean.rejected_repayments)
  -- Payments after the snapshot cannot exist in a point-in-time extract.
  AND r.repayment_date <= getvariable('snapshot_date');
