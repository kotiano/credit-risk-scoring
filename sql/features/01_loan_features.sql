-- =====================================================================================
-- features.loan_features
-- Source : clean.loans, clean.borrowers, clean.repayments, clean.transactions,
--          marts.loan_payment_events, marts.loan_repayment_summary (event dates only)
-- Grain  : one row per loan application (loan_id), i.e. one scoring event
--
-- POINT-IN-TIME CONTRACT
--   decision_date = disbursement_date of the loan being scored
--   as_of_date    = decision_date - 1 (end of the previous day)
--   Every feature uses only events dated <= as_of_date:
--     * prior loans     : disbursed <= as_of_date (same-day loans are NOT prior)
--     * repayments      : repayment_date <= as_of_date (ASOF join on cumulative paid)
--     * transactions    : transaction_date <= as_of_date
--     * prior-loan DPD  : re-computed as of as_of_date, never the loan's final outcome
--   Final outcomes (loan_state, max_dpd_to_date, bad_30dpd, source status) are never
--   read here. tests/test_features.py proves the contract by recomputing features on a
--   copy of the data truncated at the decision date and requiring identical values.
-- =====================================================================================
CREATE OR REPLACE TABLE features.loan_features AS
WITH params AS (
    SELECT
        getvariable('settlement_tolerance')::DOUBLE AS tol,
        getvariable('default_dpd')::INTEGER         AS default_dpd
),
applications AS (
    SELECT
        l.loan_id,
        l.borrower_id,
        l.disbursement_date                         AS decision_date,
        l.disbursement_date - 1                     AS as_of_date,
        l.product_type,
        l.loan_amount,
        l.term_days,
        l.interest_rate
    FROM clean.loans AS l
),

-- ------------------------------------------------------------------ prior loan history
prior_loans AS (
    SELECT
        a.loan_id,
        a.as_of_date,
        p.loan_id                                   AS prior_loan_id,
        p.disbursement_date                         AS prior_disbursement_date,
        p.due_date                                  AS prior_due_date,
        p.loan_amount                               AS prior_amount,
        p.total_due                                 AS prior_total_due,
        p.fee_amount + p.interest_amount            AS prior_charges
    FROM applications AS a
    JOIN clean.loans AS p
        ON p.borrower_id = a.borrower_id
       AND p.disbursement_date <= a.as_of_date
),
prior_position AS (
    -- Cumulative repayment of each prior loan at the end of as_of_date.
    SELECT
        pl.*,
        coalesce(e.cumulative_paid, 0)              AS paid_asof,
        coalesce(e.cumulative_payments, 0)          AS n_payments_asof
    FROM prior_loans AS pl
    ASOF LEFT JOIN marts.loan_payment_events AS e
        ON pl.prior_loan_id = e.loan_id AND pl.as_of_date >= e.repayment_date
),
prior_state AS (
    SELECT
        pp.*,
        pp.prior_total_due - pp.paid_asof <= prm.tol                AS settled_asof,
        pp.prior_due_date <= pp.as_of_date                          AS matured_asof,
        -- payoff_date is an event date; it is only used when it is <= as_of_date.
        CASE WHEN pp.prior_total_due - pp.paid_asof <= prm.tol THEN s.payoff_date END
                                                                    AS payoff_asof,
        -- Amount repaid by the contractual due date; only read for matured loans,
        -- where due_date <= as_of_date so the value is already known.
        CASE WHEN pp.prior_due_date <= pp.as_of_date THEN s.paid_by_due_date END
                                                                    AS paid_by_due_asof
    FROM prior_position AS pp
    JOIN marts.loan_repayment_summary AS s
        ON pp.prior_loan_id = s.loan_id
    CROSS JOIN params AS prm
),
prior_measured AS (
    SELECT
        ps.*,
        CASE
            WHEN ps.settled_asof OR ps.as_of_date <= ps.prior_due_date THEN 0
            ELSE date_diff('day', ps.prior_due_date, ps.as_of_date)
        END                                                         AS dpd_asof,
        greatest(date_diff('day', ps.prior_due_date,
                           CASE WHEN ps.settled_asof THEN ps.payoff_asof ELSE ps.as_of_date END),
                 0)                                                 AS max_dpd_asof,
        CASE WHEN ps.settled_asof
             THEN date_diff('day', ps.prior_due_date, ps.payoff_asof) END AS timing_days,
        CASE WHEN ps.settled_asof
             THEN date_diff('day', ps.prior_disbursement_date, ps.payoff_asof) END
                                                                    AS days_to_repay,
        CASE WHEN ps.settled_asof THEN 0
             ELSE ps.prior_amount
                  - least(greatest(ps.paid_asof - ps.prior_charges, 0), ps.prior_amount)
        END                                                         AS principal_outstanding_asof
    FROM prior_state AS ps
),
history AS (
    SELECT
        pm.loan_id,
        count(*)                                                    AS number_of_loans,
        sum(pm.prior_amount)                                        AS total_disbursed,
        avg(pm.prior_amount)                                        AS average_loan_amount,
        max(pm.prior_amount)                                        AS largest_loan,
        count(*) FILTER (WHERE NOT pm.settled_asof)                 AS active_loans,
        count(*) FILTER (WHERE pm.settled_asof)                     AS completed_loans,
        count(*) FILTER (WHERE pm.max_dpd_asof >= prm.default_dpd)  AS default_count,
        count(*) FILTER (WHERE pm.max_dpd_asof >= 30)               AS dpd_30_count,
        count(*) FILTER (WHERE pm.max_dpd_asof >= 60)               AS dpd_60_count,
        max(pm.max_dpd_asof)                                        AS max_dpd,
        avg(pm.max_dpd_asof) FILTER (WHERE pm.matured_asof)         AS avg_dpd,
        max(pm.dpd_asof)                                            AS current_dpd,
        sum(pm.principal_outstanding_asof)                          AS outstanding_principal,
        sum(pm.paid_asof) FILTER (WHERE pm.matured_asof)
            / nullif(sum(pm.prior_total_due) FILTER (WHERE pm.matured_asof), 0)
                                                                    AS repayment_rate,
        sum(pm.paid_by_due_asof) FILTER (WHERE pm.matured_asof)
            / nullif(sum(pm.prior_total_due) FILTER (WHERE pm.matured_asof), 0)
                                                                    AS collection_rate,
        coalesce(sum(pm.prior_total_due - pm.paid_asof) FILTER (WHERE pm.matured_asof), 0)
                                                                    AS collection_gap,
        avg(pm.days_to_repay)                                       AS average_days_to_repay,
        median(pm.days_to_repay)                                    AS median_days_to_repay,
        avg((pm.timing_days > 0)::INTEGER) FILTER (WHERE pm.settled_asof)
                                                                    AS late_payment_ratio,
        stddev_samp(pm.timing_days)                                 AS repayment_consistency,
        (sum(pm.n_payments_asof) - count(*) FILTER (WHERE pm.settled_asof))
            / nullif(sum(pm.n_payments_asof), 0)                    AS partial_payment_ratio,
        max(CASE
                WHEN pm.dpd_asof > 0 THEN pm.as_of_date
                WHEN pm.settled_asof AND pm.timing_days > 0 THEN pm.payoff_asof
            END)                                                    AS last_delinquent_date,
        max(pm.prior_disbursement_date)                             AS last_loan_date,
        count(*) FILTER (WHERE pm.prior_disbursement_date > pm.as_of_date - 90)
                                                                    AS borrowing_frequency_90d
    FROM prior_measured AS pm
    CROSS JOIN params AS prm
    GROUP BY pm.loan_id
),

-- ------------------------------------------------------------------ repayment recency
repayment_activity AS (
    SELECT
        a.loan_id,
        coalesce(sum(r.amount) FILTER (WHERE r.repayment_date > a.as_of_date - 30), 0)
                                                                    AS collections_last_30d,
        coalesce(sum(r.amount) FILTER (WHERE r.repayment_date > a.as_of_date - 90), 0)
                                                                    AS collections_last_90d,
        max(r.repayment_date)                                       AS last_repayment_date
    FROM applications AS a
    JOIN clean.repayments AS r
        ON r.borrower_id = a.borrower_id
       AND r.repayment_date <= a.as_of_date
    GROUP BY a.loan_id
),

-- ------------------------------------------------------------------ wallet transactions
txn_window AS (
    SELECT
        a.loan_id,
        a.as_of_date,
        t.transaction_date,
        t.direction,
        t.abs_amount,
        t.balance,
        date_diff('day', t.transaction_date, a.as_of_date) // 30   AS month_idx  -- 0 = latest
    FROM applications AS a
    JOIN clean.transactions AS t
        ON t.borrower_id = a.borrower_id
       AND t.transaction_date <= a.as_of_date
       AND t.transaction_date > a.as_of_date - 180
),
txn_90d AS (
    SELECT
        loan_id,
        count(*) FILTER (WHERE month_idx < 3)                                  AS transaction_count_90d,
        coalesce(sum(abs_amount) FILTER (WHERE month_idx < 3), 0)              AS transaction_volume_90d,
        avg(abs_amount) FILTER (WHERE month_idx < 3)                           AS average_transaction_amount_90d,
        stddev_samp(abs_amount) FILTER (WHERE month_idx < 3)
            / nullif(avg(abs_amount) FILTER (WHERE month_idx < 3), 0)          AS transaction_volatility_90d,
        coalesce(sum(abs_amount) FILTER (WHERE month_idx < 3 AND direction = 'inflow'), 0)
                                                                               AS inflow_90d,
        coalesce(sum(abs_amount) FILTER (WHERE month_idx < 3 AND direction = 'outflow'), 0)
                                                                               AS outflow_90d,
        avg(balance) FILTER (WHERE month_idx < 3)                              AS average_balance_90d,
        max(transaction_date)                                                  AS last_transaction_date
    FROM txn_window
    GROUP BY loan_id
),
monthly_inflows AS (
    -- Zero-filled 30-day inflow buckets over the last 180 days, restricted to buckets
    -- that lie entirely after the wallet account was opened.
    SELECT
        a.loan_id,
        m.idx                                                                  AS month_idx,
        coalesce(sum(w.abs_amount) FILTER (WHERE w.direction = 'inflow'), 0)   AS inflow
    FROM applications AS a
    JOIN clean.borrowers AS b USING (borrower_id)
    CROSS JOIN range(0, 6) AS m(idx)
    LEFT JOIN txn_window AS w
        ON w.loan_id = a.loan_id AND w.month_idx = m.idx
    WHERE a.as_of_date - (30 * (m.idx + 1) - 1)::INTEGER >= b.account_open_date
    GROUP BY a.loan_id, m.idx
),
income_dynamics AS (
    SELECT
        loan_id,
        count(*)                                                               AS inflow_months_observed,
        stddev_pop(inflow) / nullif(avg(inflow), 0)                            AS income_volatility_180d,
        max(inflow) FILTER (WHERE month_idx = 0)
            / nullif(avg(inflow) FILTER (WHERE month_idx IN (1, 2)), 0)        AS inflow_trend_30d
    FROM monthly_inflows
    GROUP BY loan_id
)

SELECT
    a.loan_id,
    a.borrower_id,
    a.decision_date,
    a.as_of_date,
    -- application attributes (known at decision time)
    a.product_type,
    a.loan_amount,
    a.term_days,
    a.interest_rate,
    b.employment_status,
    b.monthly_income_clean                                                     AS declared_monthly_income,
    date_diff('day', b.account_open_date, a.as_of_date)                        AS account_age_days,
    b.bureau_previous_loans,
    b.bureau_previous_defaults,
    b.pre_onboarding_txn_activity,
    a.loan_amount / nullif(b.monthly_income_clean, 0)                          AS loan_to_declared_income,
    -- loan history (as of cut-off)
    coalesce(h.number_of_loans, 0)                                             AS number_of_loans,
    coalesce(h.number_of_loans, 0) > 0                                         AS is_repeat_borrower,
    coalesce(h.total_disbursed, 0)                                             AS total_disbursed,
    h.average_loan_amount,
    h.largest_loan,
    a.loan_amount / nullif(h.largest_loan, 0)                                  AS amount_to_largest_prior,
    coalesce(h.active_loans, 0)                                                AS active_loans,
    coalesce(h.completed_loans, 0)                                             AS completed_loans,
    coalesce(h.default_count, 0)                                               AS default_count,
    coalesce(h.outstanding_principal, 0)                                       AS outstanding_principal,
    -- repayment behaviour
    h.repayment_rate,
    h.average_days_to_repay,
    h.median_days_to_repay,
    h.late_payment_ratio,
    h.repayment_consistency,
    h.partial_payment_ratio,
    -- delinquency
    coalesce(h.max_dpd, 0)                                                     AS max_dpd,
    h.avg_dpd,
    coalesce(h.current_dpd, 0)                                                 AS current_dpd,
    coalesce(h.dpd_30_count, 0)                                                AS dpd_30_count,
    coalesce(h.dpd_60_count, 0)                                                AS dpd_60_count,
    date_diff('day', h.last_delinquent_date, a.as_of_date)                     AS days_since_last_delinquency,
    -- collections
    h.collection_rate,
    round(coalesce(h.collection_gap, 0), 2)                                    AS collection_gap,
    coalesce(ra.collections_last_30d, 0)                                       AS collections_last_30d,
    coalesce(ra.collections_last_90d, 0)                                       AS collections_last_90d,
    -- transaction behaviour
    coalesce(t.transaction_count_90d, 0)                                       AS transaction_count_90d,
    coalesce(t.transaction_volume_90d, 0)                                      AS transaction_volume_90d,
    t.average_transaction_amount_90d,
    t.transaction_volatility_90d,
    coalesce(t.inflow_90d, 0) / 3.0                                            AS income_proxy_monthly,
    t.inflow_90d / nullif(t.outflow_90d, 0)                                    AS cashflow_ratio_90d,
    t.average_balance_90d,
    a.loan_amount / nullif(t.inflow_90d / 3.0, 0)                              AS loan_to_income_proxy,
    -- recency
    date_diff('day', h.last_loan_date, a.as_of_date)                           AS days_since_last_loan,
    date_diff('day', ra.last_repayment_date, a.as_of_date)                     AS days_since_last_repayment,
    date_diff('day', t.last_transaction_date, a.as_of_date)                    AS days_since_last_transaction,
    -- stability
    i.income_volatility_180d,
    CASE WHEN i.income_volatility_180d IS NOT NULL
         THEN greatest(1 - i.income_volatility_180d, 0) END                    AS income_stability_180d,
    i.inflow_trend_30d,
    coalesce(i.inflow_months_observed, 0)                                      AS inflow_months_observed,
    coalesce(h.borrowing_frequency_90d, 0)                                     AS borrowing_frequency_90d
FROM applications AS a
JOIN clean.borrowers AS b USING (borrower_id)
LEFT JOIN history AS h USING (loan_id)
LEFT JOIN repayment_activity AS ra USING (loan_id)
LEFT JOIN txn_90d AS t USING (loan_id)
LEFT JOIN income_dynamics AS i USING (loan_id);
