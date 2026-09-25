-- =====================================================================================
-- marts.loan_repayment_summary
-- Source : clean.loans, clean.repayments, marts.loan_payment_events
-- Grain  : one row per loan, measured AS OF getvariable('snapshot_date') (end of day)
--
-- Key definitions (docs/metrics.md):
--   outstanding_total     total_due - repayments to date (floored at 0)
--   is_settled            outstanding_total <= settlement_tolerance
--   payoff_date           first date cumulative repayments reach total_due - tolerance
--   principal waterfall   repayments settle fees, then interest, then principal
--   dpd                   contractual, loan-level days past due at the snapshot:
--                         snapshot - due_date while unsettled and past due, else 0
--   max_dpd_to_date       highest DPD the loan ever reached up to the snapshot:
--                         days from due_date to min(payoff_date, snapshot), floored at 0
--   loan_state            repaid | active (not yet due) | overdue | written_off (dpd >= 180)
--   bad_30dpd             model target: max_dpd_to_date >= target_dpd; only observable
--                         once snapshot >= due_date + target_dpd (performance window)
-- =====================================================================================
CREATE OR REPLACE TABLE marts.loan_repayment_summary AS
WITH params AS (
    SELECT
        getvariable('snapshot_date')::DATE              AS as_of,
        getvariable('settlement_tolerance')::DOUBLE     AS tol,
        getvariable('write_off_dpd')::INTEGER           AS write_off_dpd,
        getvariable('default_dpd')::INTEGER             AS default_dpd,
        getvariable('target_dpd')::INTEGER              AS target_dpd,
        getvariable('early_repayment_min_days')::INTEGER AS early_days
),
payment_totals AS (
    SELECT
        r.loan_id,
        sum(r.amount)                                                  AS total_paid,
        count(*)                                                       AS n_payments,
        min(r.repayment_date)                                          AS first_payment_date,
        max(r.repayment_date)                                          AS last_payment_date,
        sum(r.amount) FILTER (WHERE r.repayment_date <= l.due_date)    AS paid_by_due_date,
        sum(r.amount) FILTER (
            WHERE r.repayment_date <= l.due_date + getvariable('target_dpd')::INTEGER
        )                                                              AS paid_within_30d_of_due,
        count(*) FILTER (WHERE r.repayment_date > l.due_date)          AS n_late_payments
    FROM clean.repayments AS r
    JOIN clean.loans AS l USING (loan_id)
    GROUP BY r.loan_id
),
payoff AS (
    SELECT
        e.loan_id,
        min(e.repayment_date) AS payoff_date
    FROM marts.loan_payment_events AS e
    JOIN clean.loans AS l USING (loan_id)
    CROSS JOIN params AS p
    WHERE e.cumulative_paid >= l.total_due - p.tol
    GROUP BY e.loan_id
),
base AS (
    SELECT
        l.*,
        coalesce(pt.total_paid, 0)              AS total_paid,
        coalesce(pt.n_payments, 0)              AS n_payments,
        coalesce(pt.n_late_payments, 0)         AS n_late_payments,
        pt.first_payment_date,
        pt.last_payment_date,
        coalesce(pt.paid_by_due_date, 0)        AS paid_by_due_date,
        coalesce(pt.paid_within_30d_of_due, 0)  AS paid_within_30d_of_due,
        po.payoff_date,
        greatest(l.total_due - coalesce(pt.total_paid, 0), 0) AS outstanding_total
    FROM clean.loans AS l
    LEFT JOIN payment_totals AS pt USING (loan_id)
    LEFT JOIN payoff AS po USING (loan_id)
),
measured AS (
    SELECT
        b.*,
        b.payoff_date IS NOT NULL                                            AS is_settled,
        -- Waterfall: fees -> interest -> principal
        least(greatest(b.total_paid - b.fee_amount - b.interest_amount, 0), b.loan_amount)
                                                                             AS principal_paid,
        CASE
            WHEN b.payoff_date IS NOT NULL OR p.as_of <= b.due_date THEN 0
            ELSE date_diff('day', b.due_date, p.as_of)
        END                                                                  AS dpd,
        greatest(
            date_diff('day', b.due_date, coalesce(b.payoff_date, p.as_of)), 0
        )                                                                    AS max_dpd_to_date,
        p.as_of >= b.due_date + p.target_dpd                                 AS is_target_observable,
        p.as_of >= b.due_date + p.default_dpd                                AS is_default_observable
    FROM base AS b
    CROSS JOIN params AS p
)
SELECT
    m.loan_id,
    m.borrower_id,
    m.product_type,
    m.borrower_loan_sequence,
    m.borrower_loan_sequence = 1                                             AS is_first_loan,
    m.disbursement_date,
    date_trunc('month', m.disbursement_date)::DATE                           AS disbursement_month,
    m.due_date,
    date_trunc('month', m.due_date)::DATE                                    AS due_month,
    m.term_days,
    m.interest_rate,
    m.loan_amount,
    m.interest_amount,
    m.fee_amount,
    m.total_due,
    m.total_paid,
    m.n_payments,
    m.n_late_payments,
    m.first_payment_date,
    m.last_payment_date,
    m.paid_by_due_date,
    m.paid_within_30d_of_due,
    m.payoff_date,
    m.is_settled,
    CASE WHEN m.is_settled THEN 0 ELSE m.outstanding_total END               AS outstanding_total,
    m.principal_paid,
    CASE WHEN m.is_settled THEN 0 ELSE m.loan_amount - m.principal_paid END  AS outstanding_principal,
    m.dpd,
    dpd_bucket(m.dpd)                                                        AS dpd_bucket,
    m.max_dpd_to_date,
    dpd_bucket(m.max_dpd_to_date)                                            AS worst_dpd_bucket,
    CASE
        WHEN m.is_settled THEN 'repaid'
        WHEN m.dpd >= p.write_off_dpd THEN 'written_off'
        WHEN m.dpd > 0 THEN 'overdue'
        ELSE 'active'
    END                                                                      AS loan_state,
    m.source_loan_status,
    CASE WHEN m.is_settled THEN date_diff('day', m.disbursement_date, m.payoff_date) END
                                                                             AS days_to_repay,
    -- Negative = paid before due date
    CASE WHEN m.is_settled THEN date_diff('day', m.due_date, m.payoff_date) END
                                                                             AS repayment_timing_days,
    CASE
        WHEN NOT m.is_settled THEN NULL
        WHEN m.payoff_date <= m.due_date - p.early_days THEN 'early'
        WHEN m.payoff_date <= m.due_date THEN 'on_time'
        ELSE 'late'
    END                                                                      AS repayment_timing,
    m.n_payments > 1 OR (NOT m.is_settled AND m.n_payments > 0)              AS has_partial_payment,
    -- Default: loan reached default_dpd; the date it crossed the threshold.
    m.max_dpd_to_date >= p.default_dpd                                       AS is_default,
    CASE WHEN m.max_dpd_to_date >= p.default_dpd THEN m.due_date + p.default_dpd END
                                                                             AS default_date,
    m.max_dpd_to_date >= p.write_off_dpd                                     AS is_written_off_ever,
    m.is_target_observable,
    m.is_default_observable,
    -- NULL while the performance window is still open (right-censored).
    CASE WHEN m.is_target_observable
         THEN (m.max_dpd_to_date >= p.target_dpd)::INTEGER END              AS bad_30dpd,
    p.as_of                                                                  AS as_of_date
FROM measured AS m
CROSS JOIN params AS p;
