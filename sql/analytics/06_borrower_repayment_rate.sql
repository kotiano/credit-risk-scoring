-- Distribution of borrower-level repayment rates (repaid / due on matured loans).
SELECT
    CASE
        WHEN repayment_rate IS NULL THEN '0. no matured loans'
        WHEN repayment_rate >= 0.999 THEN '1. fully repaid (>= 99.9%)'
        WHEN repayment_rate >= 0.90 THEN '2. 90-99.9%'
        WHEN repayment_rate >= 0.50 THEN '3. 50-90%'
        ELSE '4. < 50%'
    END                                           AS repayment_rate_band,
    count(*)                                      AS borrowers,
    round(avg(total_loans), 2)                    AS avg_loans,
    round(sum(outstanding_principal), 0)          AS outstanding_principal
FROM marts.borrower_mart
GROUP BY ALL
ORDER BY repayment_rate_band;
