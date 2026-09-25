-- Where does cash come from? Collections by delinquency age at payment and channel.
SELECT
    dpd_bucket_at_payment,
    payment_channel,
    count(*)                                                        AS payments,
    round(sum(amount), 0)                                           AS amount_collected,
    round(sum(amount) / sum(sum(amount)) OVER (PARTITION BY dpd_bucket_at_payment), 4)
                                                                    AS channel_share_in_bucket
FROM marts.repayment_facts
GROUP BY dpd_bucket_at_payment, payment_channel
ORDER BY dpd_bucket_order(dpd_bucket_at_payment), amount_collected DESC;
