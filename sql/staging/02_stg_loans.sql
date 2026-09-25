-- =====================================================================================
-- staging.stg_loans
-- Source : raw.loans
-- Grain  : one row per loan_id
-- Logic  : typing, status standardisation, derived interest component, validity flags.
--          `source_loan_status` is kept for reconciliation only; lifecycle status used
--          in analytics is re-derived from repayments in marts (docs/etl.md).
-- =====================================================================================
CREATE OR REPLACE TABLE staging.stg_loans AS
WITH typed AS (
    SELECT
        trim(loan_id)                                       AS loan_id,
        trim(borrower_id)                                   AS borrower_id,
        lower(trim(product_type))                           AS product_type,
        TRY_CAST(loan_amount AS DOUBLE)                     AS loan_amount,
        TRY_CAST(interest_rate AS DOUBLE)                   AS interest_rate,
        TRY_CAST(fees AS DOUBLE)                            AS fee_amount,
        TRY_CAST(total_due AS DOUBLE)                       AS total_due,
        TRY_CAST(term_days AS INTEGER)                      AS term_days,
        TRY_CAST(disbursement_date AS DATE)                 AS disbursement_date,
        TRY_CAST(due_date AS DATE)                          AS due_date,
        lower(trim(loan_status))                            AS source_loan_status
    FROM raw.loans
),
deduplicated AS (
    SELECT
        *,
        count(*) OVER (PARTITION BY loan_id)                AS source_row_count,
        row_number() OVER (PARTITION BY loan_id ORDER BY disbursement_date) AS row_rank
    FROM typed
)
SELECT
    loan_id,
    borrower_id,
    product_type,
    loan_amount,
    interest_rate,
    fee_amount,
    round(total_due - loan_amount - fee_amount, 2)          AS interest_amount,
    total_due,
    term_days,
    disbursement_date,
    due_date,
    source_loan_status,
    source_row_count,
    coalesce(loan_amount > 0, FALSE)                        AS is_amount_valid,
    coalesce(interest_rate BETWEEN 0 AND 0.30, FALSE)       AS is_rate_valid,
    coalesce(due_date >= disbursement_date, FALSE)          AS is_date_order_valid,
    -- total_due must reconcile to principal + flat interest + fees (to the cent).
    coalesce(
        abs(total_due - (loan_amount + round(loan_amount * interest_rate * term_days / 30, 2)
                         + fee_amount)) <= 0.02,
        FALSE
    )                                                       AS is_amount_reconciled
FROM deduplicated
WHERE row_rank = 1;
