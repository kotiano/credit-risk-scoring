-- Repayment timing by product: completion, early, on-time and late repayment rates.
-- Early/on-time/late shares are among settled loans; completion among matured loans.
SELECT
    product_type,
    count(*) FILTER (WHERE due_date <= as_of_date)                               AS matured_loans,
    round(avg(is_settled::INTEGER) FILTER (WHERE due_date <= as_of_date), 4)     AS completion_rate,
    round(avg((repayment_timing = 'early')::INTEGER) FILTER (WHERE is_settled), 4)   AS early_rate,
    round(avg((repayment_timing = 'on_time')::INTEGER) FILTER (WHERE is_settled), 4) AS on_time_rate,
    round(avg((repayment_timing = 'late')::INTEGER) FILTER (WHERE is_settled), 4)    AS late_rate,
    round(avg(has_partial_payment::INTEGER), 4)                                   AS partial_payment_rate
FROM marts.loan_repayment_summary
GROUP BY product_type
ORDER BY late_rate;
