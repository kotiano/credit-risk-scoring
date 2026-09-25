-- Portfolio at Risk by month-end (share of outstanding principal at or above n DPD;
-- written-off loans excluded from numerator and denominator).
SELECT
    month_end,
    round(outstanding_principal, 0)   AS outstanding_principal,
    active_loans,
    round(par1, 4)                    AS par1,
    round(par7, 4)                    AS par7,
    round(par30, 4)                   AS par30,
    round(par60, 4)                   AS par60,
    round(par90, 4)                   AS par90
FROM marts.portfolio_monthly
ORDER BY month_end;
