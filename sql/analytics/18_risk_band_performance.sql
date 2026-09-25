-- Analytical risk bands: population, predicted PD and realised bad rate by sample window.
SELECT
    sc.sample_split,
    sc.risk_band,
    count(*)                                                                AS loans,
    min(sc.score)                                                           AS min_score,
    max(sc.score)                                                           AS max_score,
    round(avg(sc.pd), 4)                                                    AS avg_predicted_pd,
    round(avg(s.bad_30dpd) FILTER (WHERE s.is_target_observable), 4)        AS actual_bad_rate
FROM model.loan_scores AS sc
JOIN marts.loan_repayment_summary AS s USING (loan_id)
WHERE sc.sample_split IN ('train', 'validation', 'test', 'recent')
GROUP BY sc.sample_split, sc.risk_band
ORDER BY
    CASE sc.sample_split WHEN 'train' THEN 1 WHEN 'validation' THEN 2
                         WHEN 'test' THEN 3 ELSE 4 END,
    sc.risk_band;
