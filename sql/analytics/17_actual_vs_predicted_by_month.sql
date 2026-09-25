-- Model tracking: predicted PD vs actual 30+ DPD rate by decision month (champion).
-- Only matured loans have an actual outcome; maturity_share shows completeness.
SELECT
    date_trunc('month', s.disbursement_date)::DATE                          AS decision_month,
    count(*)                                                                AS scored_loans,
    round(avg(sc.pd), 4)                                                    AS avg_predicted_pd,
    count(*) FILTER (WHERE s.is_target_observable)                          AS matured_loans,
    round(avg(s.bad_30dpd) FILTER (WHERE s.is_target_observable), 4)        AS actual_bad_rate,
    round(avg(s.bad_30dpd) FILTER (WHERE s.is_target_observable)
          / avg(sc.pd) FILTER (WHERE s.is_target_observable), 3)            AS actual_to_predicted
FROM model.loan_scores AS sc
JOIN marts.loan_repayment_summary AS s USING (loan_id)
GROUP BY ALL
ORDER BY decision_month;
