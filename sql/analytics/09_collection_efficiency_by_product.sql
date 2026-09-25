-- Collection efficiency by product for amounts that fell due (expected vs actual).
SELECT
    product_type,
    sum(loans_due)                                                    AS loans_due,
    round(sum(expected_collections), 0)                               AS expected_collections,
    round(sum(collected_by_due_date) / sum(expected_collections), 4)  AS efficiency_on_time,
    round(sum(collected_within_30d) / sum(expected_collections), 4)   AS efficiency_30d,
    round(sum(collected_to_date) / sum(expected_collections), 4)      AS efficiency_to_date,
    round(sum(collection_gap), 0)                                     AS collection_gap
FROM marts.collections_due_cohort
GROUP BY product_type
ORDER BY efficiency_on_time DESC;
