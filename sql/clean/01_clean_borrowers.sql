-- =====================================================================================
-- clean.borrowers
-- Source : staging.stg_borrowers
-- Grain  : one row per borrower
-- Rules  : all borrowers pass (no borrower-level defect is disqualifying).
--          Treatment of declared income:
--            * missing           -> stays NULL (models impute + missing indicator)
--            * implausible (flag) -> `monthly_income_clean` set to NULL; raw value kept
--              in `monthly_income` for audit. We do not guess the "intended" value.
-- =====================================================================================
CREATE OR REPLACE TABLE clean.borrowers AS
SELECT
    borrower_id,
    onboarding_date,
    account_open_date,
    age,
    gender,
    employment_status,
    monthly_income,
    CASE WHEN is_income_outlier THEN NULL ELSE monthly_income END AS monthly_income_clean,
    is_income_missing,
    is_income_outlier,
    coalesce(region, 'unknown')                              AS region,
    bureau_previous_loans,
    bureau_previous_defaults,
    pre_onboarding_txn_activity
FROM staging.stg_borrowers;
