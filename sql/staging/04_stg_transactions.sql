-- =====================================================================================
-- staging.stg_transactions
-- Source : raw.transactions (wallet activity; lending flows are NOT in this feed)
-- Grain  : one row per transaction_id
-- Logic  : typing, label standardisation, dedup, outlier and validity flags.
-- =====================================================================================
CREATE OR REPLACE TABLE staging.stg_transactions AS
WITH typed AS (
    SELECT
        trim(transaction_id)                                AS transaction_id,
        trim(borrower_id)                                   AS borrower_id,
        TRY_CAST(transaction_date AS DATE)                  AS transaction_date,
        lower(trim(transaction_type))                       AS transaction_type,
        TRY_CAST(amount AS DOUBLE)                          AS amount,
        TRY_CAST(balance AS DOUBLE)                         AS balance,
        lower(trim(channel))                                AS channel
    FROM raw.transactions
),
deduplicated AS (
    SELECT
        *,
        count(*) OVER (PARTITION BY transaction_id)         AS source_row_count,
        row_number() OVER (PARTITION BY transaction_id ORDER BY transaction_date) AS row_rank
    FROM typed
),
borrower_scale AS (
    -- Per-borrower robust scale: an amount is only "suspicious" relative to that
    -- borrower's own typical activity (median/MAD of log absolute amounts).
    SELECT
        borrower_id,
        quantile_cont(ln(abs(amount)), 0.25)                AS q1_log_amount,
        quantile_cont(ln(abs(amount)), 0.75)                AS q3_log_amount
    FROM deduplicated
    WHERE row_rank = 1 AND amount <> 0
    GROUP BY borrower_id
)
SELECT
    t.transaction_id,
    t.borrower_id,
    t.transaction_date,
    t.transaction_type,
    CASE WHEN t.amount >= 0 THEN 'inflow' ELSE 'outflow' END AS direction,
    t.amount,
    t.balance,
    t.channel,
    t.source_row_count,
    t.amount IS NOT NULL                                    AS is_amount_present,
    coalesce(t.transaction_date <= getvariable('snapshot_date'), FALSE) AS is_date_valid,
    b.borrower_id IS NOT NULL                               AS has_parent_borrower,
    -- Upper Tukey fence (Q3 + 2 x IQR) on the log scale, per borrower. Flag only:
    -- large legitimate transfers exist, so outliers are reviewed, not deleted.
    coalesce(
        ln(abs(t.amount)) > s.q3_log_amount + 2 * (s.q3_log_amount - s.q1_log_amount),
        FALSE
    )                                                       AS is_amount_outlier
FROM deduplicated AS t
LEFT JOIN staging.stg_borrowers AS b
    ON t.borrower_id = b.borrower_id
LEFT JOIN borrower_scale AS s
    ON t.borrower_id = s.borrower_id
WHERE t.row_rank = 1;
