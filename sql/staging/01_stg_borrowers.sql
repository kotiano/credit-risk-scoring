-- =====================================================================================
-- staging.stg_borrowers
-- Source : raw.borrowers (all VARCHAR, as extracted)
-- Grain  : one row per borrower_id (exact duplicate rows collapsed)
-- Logic  : type casting, label standardisation, duplicate collapse, quality flags.
--          Rows are flagged here, never silently dropped; `clean` decides what passes.
-- =====================================================================================
CREATE OR REPLACE TABLE staging.stg_borrowers AS
WITH typed AS (
    SELECT
        trim(borrower_id)                                   AS borrower_id,
        TRY_CAST(onboarding_date AS DATE)                   AS onboarding_date,
        TRY_CAST(account_open_date AS DATE)                 AS account_open_date,
        TRY_CAST(age AS INTEGER)                            AS age,
        -- Source mixes codes ('F') and words ('female'); standardise on the first letter.
        CASE upper(left(trim(gender), 1))
            WHEN 'F' THEN 'female'
            WHEN 'M' THEN 'male'
            ELSE 'undisclosed'
        END                                                 AS gender,
        lower(trim(employment_status))                      AS employment_status,
        TRY_CAST(monthly_income AS DOUBLE)                  AS monthly_income,
        nullif(trim(region), '')                            AS region,
        TRY_CAST(previous_loans AS INTEGER)                 AS bureau_previous_loans,
        TRY_CAST(previous_defaults AS INTEGER)              AS bureau_previous_defaults,
        TRY_CAST(transaction_activity AS INTEGER)           AS pre_onboarding_txn_activity
    FROM raw.borrowers
),
deduplicated AS (
    SELECT
        *,
        count(*) OVER (PARTITION BY borrower_id)            AS source_row_count,
        row_number() OVER (PARTITION BY borrower_id ORDER BY onboarding_date) AS row_rank
    FROM typed
),
income_stats AS (
    -- Robust scale (median / MAD of log income) within each employment group, because
    -- plausible income ranges differ sharply between e.g. salaried and student borrowers.
    SELECT
        employment_status,
        median(log_income)                                  AS med_log_income,
        median(abs(log_income - grp_median))                AS mad_log_income
    FROM (
        SELECT
            employment_status,
            ln(monthly_income)                              AS log_income,
            median(ln(monthly_income)) OVER (PARTITION BY employment_status) AS grp_median
        FROM typed
        WHERE monthly_income > 0
    )
    GROUP BY employment_status
)
SELECT
    d.borrower_id,
    d.onboarding_date,
    d.account_open_date,
    d.age,
    d.gender,
    d.employment_status,
    d.monthly_income,
    d.region,
    d.bureau_previous_loans,
    d.bureau_previous_defaults,
    d.pre_onboarding_txn_activity,
    d.source_row_count,
    d.monthly_income IS NULL                                AS is_income_missing,
    -- Robust z-score > 5 on log income: implausible for declared monthly income
    -- (typically a unit/keying error). Flagged, not deleted.
    coalesce(
        abs(ln(d.monthly_income) - s.med_log_income) / nullif(1.4826 * s.mad_log_income, 0) > 5,
        FALSE
    )                                                       AS is_income_outlier
FROM deduplicated AS d
LEFT JOIN income_stats AS s
    ON d.employment_status = s.employment_status
WHERE d.row_rank = 1;
