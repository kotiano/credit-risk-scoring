-- Reconciliation: source-system loan status vs status re-derived from clean repayments.
-- Mismatches arise where quarantined payments changed the balance; routed to the source
-- owner rather than silently "fixed".
SELECT
    source_loan_status,
    loan_state                              AS derived_loan_state,
    source_loan_status = loan_state         AS is_match,
    count(*)                                AS loans
FROM marts.loan_repayment_summary
GROUP BY source_loan_status, loan_state
ORDER BY is_match DESC, loans DESC;
