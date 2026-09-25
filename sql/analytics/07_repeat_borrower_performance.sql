-- Performance by loan sequence: do repeat borrowers perform better?
-- Outcome rates use matured loans only (performance window complete).
SELECT
    CASE WHEN borrower_loan_sequence >= 5 THEN '5+'
         ELSE borrower_loan_sequence::VARCHAR END                    AS loan_sequence,
    count(*)                                                        AS loans,
    round(avg(loan_amount), 0)                                      AS avg_loan_amount,
    round(avg(bad_30dpd) FILTER (WHERE is_target_observable), 4)    AS bad_30dpd_rate,
    round(avg(is_default::INTEGER) FILTER (WHERE is_default_observable), 4) AS default_rate,
    round(avg((repayment_timing = 'early')::INTEGER) FILTER (WHERE is_settled), 4)
                                                                    AS early_repayment_rate
FROM marts.loan_repayment_summary
GROUP BY ALL
ORDER BY min(borrower_loan_sequence);
