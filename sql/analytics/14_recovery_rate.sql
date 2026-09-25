-- Post-default recovery by default cohort (quarter) and product.
SELECT
    date_trunc('quarter', default_month)::DATE                          AS default_quarter,
    product_type,
    count(*)                                                            AS defaulted_loans,
    round(sum(exposure_at_default), 0)                                  AS exposure_at_default,
    round(sum(recovered_amount), 0)                                     AS recovered,
    round(sum(recovered_amount) / nullif(sum(exposure_at_default), 0), 4) AS recovery_rate
FROM marts.loan_default_recovery
GROUP BY ALL
ORDER BY default_quarter, product_type;
