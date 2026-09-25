-- Default and delinquency outcomes by rule-based borrower segment (as of snapshot).
-- Descriptive only: segments are assigned with hindsight, so this is not a forecast.
SELECT
    bm.risk_segment,
    count(DISTINCT bm.borrower_id)                                           AS borrowers,
    count(*)                                                                 AS loans,
    round(avg(s.bad_30dpd) FILTER (WHERE s.is_target_observable), 4)         AS bad_30dpd_rate,
    round(avg(s.is_default::INTEGER) FILTER (WHERE s.is_default_observable), 4) AS default_rate,
    round(sum(s.outstanding_principal) FILTER (WHERE s.loan_state IN ('active', 'overdue')), 0)
                                                                             AS outstanding_principal
FROM marts.borrower_mart AS bm
JOIN marts.loan_repayment_summary AS s USING (borrower_id)
GROUP BY bm.risk_segment
ORDER BY default_rate;
