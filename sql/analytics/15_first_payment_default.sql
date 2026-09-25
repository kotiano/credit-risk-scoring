-- First-payment default (FPD30): a borrower's FIRST loan reaches 30+ DPD.
-- For single-instalment loans the first payment is the only scheduled payment.
SELECT
    disbursement_month,
    count(*) FILTER (WHERE is_target_observable)                     AS first_loans_matured,
    round(avg(bad_30dpd) FILTER (WHERE is_target_observable), 4)     AS fpd30_rate
FROM marts.loan_repayment_summary
WHERE is_first_loan
GROUP BY disbursement_month
ORDER BY disbursement_month;
