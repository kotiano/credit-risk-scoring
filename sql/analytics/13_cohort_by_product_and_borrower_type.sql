-- Cohort performance by product and new vs repeat borrowers (matured outcomes only).
SELECT
    product_type,
    borrower_type,
    sum(loans)                                                                        AS loans,
    round(sum(amount_disbursed), 0)                                                   AS amount_disbursed,
    round(sum(bad_30dpd_rate * loans_30dpd_observable) / sum(loans_30dpd_observable), 4)
                                                                                      AS bad_30dpd_rate,
    round(sum(default_rate * loans_default_observable) / sum(loans_default_observable), 4)
                                                                                      AS default_rate,
    round(sum(total_repaid) / sum(total_due), 4)                                      AS repayment_rate_to_date
FROM marts.cohort_performance
GROUP BY ALL
ORDER BY product_type, borrower_type;
