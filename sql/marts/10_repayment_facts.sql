-- =====================================================================================
-- marts.repayment_facts
-- Source : clean.repayments, marts.loan_repayment_summary, marts.borrower_mart
-- Grain  : one row per repayment
-- Purpose: collections analysis by channel, product, region, segment and by the loan's
--          delinquency age at the time of payment (dpd_at_payment).
--   dpd_at_payment = max(0, repayment_date - due_date)  -- the loan is necessarily
--   unsettled before any payment, so this is the loan's DPD on the payment day.
-- =====================================================================================
CREATE OR REPLACE TABLE marts.repayment_facts AS
SELECT
    r.repayment_id,
    r.loan_id,
    r.borrower_id,
    r.repayment_date,
    date_trunc('month', r.repayment_date)::DATE                     AS repayment_month,
    r.amount,
    r.payment_channel,
    s.product_type,
    s.disbursement_month,
    s.due_date,
    greatest(date_diff('day', s.due_date, r.repayment_date), 0)     AS dpd_at_payment,
    dpd_bucket(greatest(date_diff('day', s.due_date, r.repayment_date), 0)) AS dpd_bucket_at_payment,
    s.default_date IS NOT NULL AND r.repayment_date > s.default_date AS is_post_default_recovery,
    bm.region,
    bm.risk_segment
FROM clean.repayments AS r
JOIN marts.loan_repayment_summary AS s USING (loan_id)
JOIN marts.borrower_mart AS bm ON r.borrower_id = bm.borrower_id;
