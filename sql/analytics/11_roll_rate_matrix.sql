-- Average month-over-month roll-rate matrix (count basis) across all months.
-- Row = bucket at month M, column = state at month M+1; rows sum to 1.
WITH transitions AS (
    SELECT from_state, to_state, sum(n_loans) AS n
    FROM marts.roll_rates
    GROUP BY ALL
),
shares AS (
    SELECT from_state, to_state, n / sum(n) OVER (PARTITION BY from_state) AS share
    FROM transitions
)
PIVOT shares
ON to_state IN ('current', '1-6', '7-29', '30-59', '60-89', '90+', 'repaid', 'written_off')
USING round(sum(share), 4)
GROUP BY from_state
ORDER BY dpd_bucket_order(from_state);
