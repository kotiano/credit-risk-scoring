-- Vintage analysis: cumulative share of each disbursement cohort that reached 30+ DPD
-- by months on book (MOB). Blank cells are not yet observable.
SELECT
    disbursement_month,
    max(loans)                                                               AS loans,
    round(max(cum_bad_30dpd_rate) FILTER (WHERE months_on_book = 1), 4)      AS mob_1,
    round(max(cum_bad_30dpd_rate) FILTER (WHERE months_on_book = 2), 4)      AS mob_2,
    round(max(cum_bad_30dpd_rate) FILTER (WHERE months_on_book = 3), 4)      AS mob_3,
    round(max(cum_bad_30dpd_rate) FILTER (WHERE months_on_book = 6), 4)      AS mob_6,
    round(max(cum_default_rate) FILTER (WHERE months_on_book = 6), 4)        AS default_mob_6
FROM marts.vintage_curves
GROUP BY disbursement_month
ORDER BY disbursement_month;
