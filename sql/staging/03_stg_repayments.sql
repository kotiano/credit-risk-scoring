-- =====================================================================================
-- staging.stg_repayments
-- Source : raw.repayments
-- Grain  : one row per payment event (unique transaction_reference)
-- Logic  : two duplicate patterns exist in the source feed:
--            1. exact duplicate rows (same repayment_id)          -> extract duplication
--            2. same transaction_reference, new repayment_id       -> integration re-post
--          Both are collapsed, keeping the earliest repayment_id. The payment reference
--          is the business key because it identifies the money movement.
-- =====================================================================================
CREATE OR REPLACE TABLE staging.stg_repayments AS
WITH typed AS (
    SELECT
        trim(repayment_id)                                  AS repayment_id,
        trim(loan_id)                                       AS loan_id,
        trim(borrower_id)                                   AS borrower_id,
        TRY_CAST(repayment_date AS DATE)                    AS repayment_date,
        TRY_CAST(amount AS DOUBLE)                          AS amount,
        -- 'Mobile App', ' mobile_app ', 'MOBILE_APP' -> 'mobile_app'
        replace(lower(trim(payment_channel)), ' ', '_')     AS payment_channel,
        trim(transaction_reference)                         AS transaction_reference
    FROM raw.repayments
),
deduplicated AS (
    SELECT
        *,
        count(*) OVER (PARTITION BY transaction_reference)  AS source_row_count,
        row_number() OVER (
            PARTITION BY transaction_reference ORDER BY repayment_id
        )                                                   AS row_rank
    FROM typed
)
SELECT
    r.repayment_id,
    r.loan_id,
    r.borrower_id,
    r.repayment_date,
    r.amount,
    r.payment_channel,
    r.transaction_reference,
    r.source_row_count,
    coalesce(r.amount > 0, FALSE)                           AS is_amount_valid,
    l.loan_id IS NOT NULL                                   AS has_parent_loan,
    coalesce(r.repayment_date >= l.disbursement_date, FALSE) AS is_date_valid,
    coalesce(r.borrower_id = l.borrower_id, FALSE)          AS is_borrower_consistent
FROM deduplicated AS r
LEFT JOIN staging.stg_loans AS l
    ON r.loan_id = l.loan_id
WHERE r.row_rank = 1;
