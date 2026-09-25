-- Monthly collections: cash collected vs amounts falling due in the month.
-- A collection rate above 100% means arrears from earlier months were recovered.
SELECT
    month_end,
    round(amount_falling_due, 0)                          AS amount_falling_due,
    round(collections, 0)                                 AS cash_collected,
    round(collection_rate, 3)                             AS collection_rate,
    round(collections - amount_falling_due, 0)            AS surplus_or_gap
FROM marts.portfolio_monthly
ORDER BY month_end;
