-- Delinquency statistics at the snapshot: delinquency rate and DPD moments of
-- delinquent on-book loans.
SELECT
    count(*) FILTER (WHERE loan_state IN ('active', 'overdue'))                 AS on_book_loans,
    count(*) FILTER (WHERE loan_state = 'overdue')                              AS delinquent_loans,
    round(count(*) FILTER (WHERE loan_state = 'overdue')
          / count(*) FILTER (WHERE loan_state IN ('active', 'overdue')), 4)     AS delinquency_rate,
    round(avg(dpd) FILTER (WHERE loan_state = 'overdue'), 1)                    AS avg_dpd,
    median(dpd) FILTER (WHERE loan_state = 'overdue')                           AS median_dpd,
    max(dpd) FILTER (WHERE loan_state = 'overdue')                              AS max_dpd
FROM marts.loan_repayment_summary;
