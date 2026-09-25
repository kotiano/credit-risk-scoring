-- =====================================================================================
-- clean.loans / clean.rejected_loans
-- Source : staging.stg_loans
-- Grain  : one row per loan
-- Rules  : a loan is quarantined when its core contract terms are unusable for
--          delinquency measurement: non-positive amount, due date before disbursement,
--          invalid rate, total_due that does not reconcile, or unknown borrower.
--          Quarantined rows are kept in clean.rejected_loans with the reason, for
--          follow-up with the source-system owner.
-- =====================================================================================
CREATE OR REPLACE TABLE clean.rejected_loans AS
SELECT
    l.*,
    concat_ws(', ',
        CASE WHEN NOT l.is_amount_valid THEN 'non_positive_amount' END,
        CASE WHEN NOT l.is_date_order_valid THEN 'due_before_disbursement' END,
        CASE WHEN NOT l.is_rate_valid THEN 'invalid_interest_rate' END,
        CASE WHEN NOT l.is_amount_reconciled THEN 'total_due_not_reconciled' END,
        CASE WHEN b.borrower_id IS NULL THEN 'unknown_borrower' END
    )                                                        AS rejection_reason
FROM staging.stg_loans AS l
LEFT JOIN clean.borrowers AS b
    ON l.borrower_id = b.borrower_id
WHERE NOT (l.is_amount_valid AND l.is_date_order_valid AND l.is_rate_valid
           AND l.is_amount_reconciled)
   OR b.borrower_id IS NULL;

CREATE OR REPLACE TABLE clean.loans AS
SELECT
    l.loan_id,
    l.borrower_id,
    l.product_type,
    l.loan_amount,
    l.interest_rate,
    l.interest_amount,
    l.fee_amount,
    l.total_due,
    l.term_days,
    l.disbursement_date,
    l.due_date,
    l.source_loan_status,
    -- Loan sequence per borrower (1 = first loan on the platform).
    row_number() OVER (
        PARTITION BY l.borrower_id ORDER BY l.disbursement_date, l.loan_id
    )                                                        AS borrower_loan_sequence
FROM staging.stg_loans AS l
WHERE l.loan_id NOT IN (SELECT loan_id FROM clean.rejected_loans);
