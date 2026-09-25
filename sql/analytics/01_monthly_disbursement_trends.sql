-- Monthly disbursement trends: volume, value, average ticket and repeat-borrower share.
SELECT
    disbursement_month,
    count(*)                                              AS loans_disbursed,
    count(DISTINCT borrower_id)                           AS borrowers,
    round(sum(loan_amount), 0)                            AS amount_disbursed,
    round(avg(loan_amount), 0)                            AS avg_loan_amount,
    round(avg((NOT is_first_loan)::INTEGER), 3)           AS repeat_loan_share,
    round(sum(loan_amount) / lag(sum(loan_amount)) OVER (ORDER BY disbursement_month) - 1, 3)
                                                          AS mom_amount_growth
FROM marts.loan_repayment_summary
GROUP BY disbursement_month
ORDER BY disbursement_month;
