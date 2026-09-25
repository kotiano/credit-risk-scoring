-- =====================================================================================
-- marts.loan_payment_events
-- Source : clean.repayments
-- Grain  : one row per loan per payment date
-- Purpose: cumulative repayment position of every loan at the end of each payment day.
--          This is the building block for all "as of date D" balances: the balance of a
--          loan at the end of day D is total_due minus the cumulative_paid of the latest
--          event on or before D (an ASOF join).
-- =====================================================================================
CREATE OR REPLACE TABLE marts.loan_payment_events AS
WITH daily AS (
    SELECT
        loan_id,
        repayment_date,
        sum(amount)     AS paid_on_date,
        count(*)        AS payments_on_date
    FROM clean.repayments
    GROUP BY loan_id, repayment_date
)
SELECT
    loan_id,
    repayment_date,
    paid_on_date,
    payments_on_date,
    sum(paid_on_date) OVER w      AS cumulative_paid,
    sum(payments_on_date) OVER w  AS cumulative_payments
FROM daily
WINDOW w AS (PARTITION BY loan_id ORDER BY repayment_date ROWS UNBOUNDED PRECEDING);
