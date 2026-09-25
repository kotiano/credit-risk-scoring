-- =====================================================================================
-- features.model_dataset
-- Source : features.loan_features, marts.loan_repayment_summary
-- Grain  : one row per loan application
--
-- Target  : bad_30dpd = 1 if the loan reached 30 days past due (unsettled on
--           due_date + 30), else 0.
-- Windows : observation window = all history up to decision_date - 1
--           performance window = disbursement -> due_date + 30 days
-- Sample  : `sample_split` assigns loans out-of-time by decision (disbursement) date:
--             train       decision_date <= train_end
--             validation  train_end      < decision_date <= validation_end
--             test        validation_end < decision_date <= test_end
--             recent      decision_date  > test_end, performance window complete
--             unobserved  performance window still open at the snapshot (censored);
--                         scored for monitoring but never used for training/evaluation
-- =====================================================================================
CREATE OR REPLACE TABLE features.model_dataset AS
SELECT
    f.*,
    s.due_date,
    s.bad_30dpd                                                     AS target,
    s.is_target_observable,
    CASE
        WHEN NOT s.is_target_observable                 THEN 'unobserved'
        WHEN f.decision_date <= getvariable('train_end')      THEN 'train'
        WHEN f.decision_date <= getvariable('validation_end') THEN 'validation'
        WHEN f.decision_date <= getvariable('test_end')       THEN 'test'
        ELSE 'recent'
    END                                                             AS sample_split
FROM features.loan_features AS f
JOIN marts.loan_repayment_summary AS s USING (loan_id);
