-- =====================================================================================
-- marts.loan_default_recovery
-- Source : marts.loan_repayment_summary, marts.loan_payment_events
-- Grain  : one row per defaulted loan (reached default_dpd)
-- Purpose: exposure at default and post-default recoveries.
--   exposure_at_default  = total_due - cumulative_paid at end of default_date
--   recovered_amount     = repayments strictly after default_date (to snapshot)
--   recovery_rate        = recovered_amount / exposure_at_default
-- =====================================================================================
CREATE OR REPLACE TABLE marts.loan_default_recovery AS
WITH defaulted AS (
    SELECT loan_id, borrower_id, product_type, disbursement_month, total_due, total_paid,
           default_date, loan_amount
    FROM marts.loan_repayment_summary
    WHERE is_default
),
paid_at_default AS (
    SELECT
        d.loan_id,
        coalesce(e.cumulative_paid, 0) AS paid_at_default
    FROM defaulted AS d
    ASOF LEFT JOIN marts.loan_payment_events AS e
        ON d.loan_id = e.loan_id AND d.default_date >= e.repayment_date
)
SELECT
    d.loan_id,
    d.borrower_id,
    d.product_type,
    d.disbursement_month,
    d.default_date,
    date_trunc('month', d.default_date)::DATE           AS default_month,
    d.loan_amount,
    d.total_due,
    d.total_due - p.paid_at_default                     AS exposure_at_default,
    d.total_paid - p.paid_at_default                    AS recovered_amount,
    (d.total_paid - p.paid_at_default)
        / nullif(d.total_due - p.paid_at_default, 0)    AS recovery_rate
FROM defaulted AS d
JOIN paid_at_default AS p USING (loan_id);
