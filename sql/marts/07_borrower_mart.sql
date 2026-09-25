-- =====================================================================================
-- marts.borrower_mart
-- Source : clean.borrowers, marts.loan_repayment_summary
-- Grain  : one row per borrower with at least one loan, AS OF the snapshot date
-- Purpose: borrower-level lending KPIs and the rule-based behavioural risk segment used
--          for portfolio monitoring. NOT a model feature table (features are built
--          point-in-time in sql/features) and NOT an underwriting decision.
--
-- Segment rules are evaluated top-down; the first matching rule wins (docs/metrics.md).
-- =====================================================================================
CREATE OR REPLACE TABLE marts.borrower_mart AS
WITH params AS (
    SELECT getvariable('snapshot_date')::DATE AS as_of
),
loan_agg AS (
    SELECT
        s.borrower_id,
        count(*)                                                        AS total_loans,
        sum(s.loan_amount)                                              AS total_disbursed,
        sum(s.total_due)                                                AS total_due,
        sum(s.total_paid)                                               AS total_repaid,
        coalesce(sum(s.outstanding_principal)
                 FILTER (WHERE s.loan_state IN ('active', 'overdue')), 0) AS outstanding_principal,
        count(*) FILTER (WHERE s.loan_state IN ('active', 'overdue'))   AS active_loans,
        count(*) FILTER (WHERE s.is_settled)                            AS completed_loans,
        count(*) FILTER (WHERE s.loan_state = 'written_off')            AS written_off_loans,
        sum(s.n_payments)                                               AS n_repayments,
        sum(s.n_late_payments)                                          AS n_late_repayments,
        count(*) FILTER (WHERE s.repayment_timing = 'late')             AS late_settled_loans,
        count(*) FILTER (WHERE s.repayment_timing = 'early')            AS early_settled_loans,
        count(*) FILTER (WHERE s.max_dpd_to_date >= 30)                 AS loans_reached_30dpd,
        count(*) FILTER (WHERE s.is_default)                            AS n_defaults,
        sum(s.total_paid) FILTER (WHERE s.due_date <= p.as_of)
            / nullif(sum(s.total_due) FILTER (WHERE s.due_date <= p.as_of), 0)
                                                                        AS repayment_rate,
        coalesce(max(s.dpd) FILTER (WHERE s.loan_state IN ('active', 'overdue')), 0)
                                                                        AS current_dpd,
        max(s.max_dpd_to_date)                                          AS max_dpd_ever,
        avg(s.days_to_repay)                                            AS avg_days_to_repay,
        avg(s.repayment_timing_days)                                    AS avg_repayment_timing_days,
        min(s.disbursement_date)                                        AS first_loan_date,
        max(s.disbursement_date)                                        AS last_loan_date,
        max(s.last_payment_date)                                        AS last_repayment_date,
        arg_max(s.max_dpd_to_date, s.payoff_date) FILTER (WHERE s.is_settled)
                                                                        AS latest_settled_max_dpd
    FROM marts.loan_repayment_summary AS s
    CROSS JOIN params AS p
    GROUP BY s.borrower_id
),
derived AS (
    SELECT
        a.*,
        a.total_repaid / nullif(a.n_repayments, 0)                      AS avg_repayment_amount,
        -- repayments per 30 days of lending relationship
        a.n_repayments / greatest(date_diff('day', a.first_loan_date, p.as_of) / 30.0, 1)
                                                                        AS repayment_frequency_30d,
        a.late_settled_loans / nullif(a.completed_loans, 0)             AS late_payment_ratio,
        date_diff('day', a.last_loan_date, p.as_of)                     AS days_since_last_loan,
        date_diff('day', a.last_repayment_date, p.as_of)                AS days_since_last_repayment
    FROM loan_agg AS a
    CROSS JOIN params AS p
)
SELECT
    b.borrower_id,
    b.region,
    b.employment_status,
    b.gender,
    CASE
        WHEN b.age < 25 THEN '18-24'
        WHEN b.age < 35 THEN '25-34'
        WHEN b.age < 45 THEN '35-44'
        WHEN b.age < 55 THEN '45-54'
        ELSE '55+'
    END                                                                 AS age_band,
    b.onboarding_date,
    d.* EXCLUDE (borrower_id),
    d.total_loans > 1                                                   AS is_repeat_borrower,
    CASE
        WHEN d.n_defaults > 0
            THEN 'Default / Recovery'
        WHEN d.current_dpd >= 30 OR d.loans_reached_30dpd >= 2
            THEN 'High Risk / Persistent Delinquency'
        WHEN d.current_dpd >= 1 OR coalesce(d.latest_settled_max_dpd, 0) >= 7
            THEN 'Emerging Delinquency'
        WHEN d.max_dpd_ever >= 7 OR coalesce(d.late_payment_ratio, 0) > 1.0 / 3
            THEN 'Moderate Risk'
        WHEN d.completed_loans <= 1
            THEN 'Low Risk / New Borrower'
        ELSE 'Low Risk / Strong Repayer'
    END                                                                 AS risk_segment
FROM derived AS d
JOIN clean.borrowers AS b USING (borrower_id);
