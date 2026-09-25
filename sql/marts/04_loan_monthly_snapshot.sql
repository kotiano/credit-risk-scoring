-- =====================================================================================
-- marts.loan_monthly_snapshot
-- Source : clean.loans, marts.loan_payment_events, marts.loan_repayment_summary
-- Grain  : one row per loan per month-end at which the loan was on book during the month
--          (disbursed on/before month-end and not settled before the month started).
-- Purpose: point-in-time balances, DPD and state at each month-end -> PAR trends,
--          roll rates, delinquency distributions. Month-end = last day of month, capped
--          at the snapshot date for the current month.
-- =====================================================================================
CREATE OR REPLACE TABLE marts.loan_monthly_snapshot AS
WITH params AS (
    SELECT
        getvariable('snapshot_date')::DATE          AS as_of,
        getvariable('settlement_tolerance')::DOUBLE AS tol,
        getvariable('write_off_dpd')::INTEGER       AS write_off_dpd
),
month_ends AS (
    SELECT
        least(last_day(m::DATE), p.as_of)                          AS month_end,
        m::DATE                                                    AS month_start
    FROM params AS p,
         generate_series(
             date_trunc('month', (SELECT min(disbursement_date) FROM clean.loans)),
             date_trunc('month', p.as_of),
             INTERVAL 1 MONTH
         ) AS g(m)
),
grid AS (
    SELECT
        s.loan_id, s.borrower_id, s.product_type, s.disbursement_date, s.due_date,
        s.loan_amount, s.interest_amount, s.fee_amount, s.total_due, s.payoff_date,
        me.month_end, me.month_start
    FROM marts.loan_repayment_summary AS s
    JOIN month_ends AS me
        ON s.disbursement_date <= me.month_end
       AND (s.payoff_date IS NULL OR s.payoff_date >= me.month_start)
),
positioned AS (
    SELECT
        g.*,
        coalesce(e.cumulative_paid, 0) AS paid_to_date
    FROM grid AS g
    ASOF LEFT JOIN marts.loan_payment_events AS e
        ON g.loan_id = e.loan_id AND g.month_end >= e.repayment_date
),
collected AS (
    SELECT
        loan_id,
        date_trunc('month', repayment_date)::DATE AS month_start,
        sum(amount)                                AS collected_in_month
    FROM clean.repayments
    GROUP BY ALL
),
measured AS (
    SELECT
        ps.*,
        coalesce(c.collected_in_month, 0)                          AS collected_in_month,
        ps.total_due - ps.paid_to_date <= p.tol                    AS is_settled,
        least(greatest(ps.paid_to_date - ps.fee_amount - ps.interest_amount, 0),
              ps.loan_amount)                                      AS principal_paid
    FROM positioned AS ps
    LEFT JOIN collected AS c USING (loan_id, month_start)
    CROSS JOIN params AS p
),
with_dpd AS (
    SELECT
        m.*,
        CASE
            WHEN m.is_settled OR m.month_end <= m.due_date THEN 0
            ELSE date_diff('day', m.due_date, m.month_end)
        END AS dpd
    FROM measured AS m
)
SELECT
    w.loan_id,
    w.borrower_id,
    w.product_type,
    w.month_end,
    w.disbursement_date,
    w.due_date,
    w.loan_amount,
    w.total_due,
    w.paid_to_date,
    w.collected_in_month,
    CASE WHEN w.is_settled THEN 0 ELSE greatest(w.total_due - w.paid_to_date, 0) END
                                                        AS outstanding_total,
    CASE WHEN w.is_settled THEN 0 ELSE w.loan_amount - w.principal_paid END
                                                        AS outstanding_principal,
    w.dpd,
    dpd_bucket(w.dpd)                                   AS dpd_bucket,
    CASE
        WHEN w.is_settled THEN 'repaid'
        WHEN w.dpd >= p.write_off_dpd THEN 'written_off'
        WHEN w.dpd > 0 THEN 'overdue'
        ELSE 'active'
    END                                                 AS loan_state
FROM with_dpd AS w
CROSS JOIN params AS p;
