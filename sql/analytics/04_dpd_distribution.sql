-- DPD distribution of the on-book portfolio at the snapshot date.
WITH on_book AS (
    SELECT dpd_bucket, dpd, outstanding_principal
    FROM marts.loan_repayment_summary
    WHERE loan_state IN ('active', 'overdue')
)
SELECT
    dpd_bucket,
    count(*)                                                        AS loans,
    round(sum(outstanding_principal), 0)                            AS outstanding_principal,
    round(sum(outstanding_principal) / sum(sum(outstanding_principal)) OVER (), 4)
                                                                    AS principal_share,
    round(avg(dpd), 1)                                              AS avg_dpd
FROM on_book
GROUP BY dpd_bucket
ORDER BY dpd_bucket_order(dpd_bucket);
