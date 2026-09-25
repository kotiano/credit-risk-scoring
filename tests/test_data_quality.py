"""Data-quality framework detects every injected defect and the clean layer is sound."""

from __future__ import annotations

from conftest import PipelineRun


def _check(run: PipelineRun, name: str) -> dict:
    return next(c for c in run.dq["checks"] if c["name"] == name)


def test_clean_layer_invariants_all_pass(pipeline: PipelineRun) -> None:
    clean = [c for c in pipeline.dq["checks"] if c["layer"] == "clean"]
    assert clean and all(c["status"] == "PASS" for c in clean)
    assert pipeline.dq["summary"]["clean_layer_status"] == "PASS"


def test_injected_defects_are_detected_exactly(pipeline: PipelineRun) -> None:
    inj = pipeline.manifest["injected_issues"]
    assert (
        _check(pipeline, "duplicate_borrower_rows")["failing_records"]
        == inj["borrowers"]["exact_duplicates"]
    )
    assert (
        _check(pipeline, "missing_declared_income")["failing_records"]
        == inj["borrowers"]["missing_income"]
    )
    assert (
        _check(pipeline, "duplicate_loan_rows")["failing_records"]
        == inj["loans"]["exact_duplicates"]
    )
    assert (
        _check(pipeline, "due_before_disbursement")["failing_records"]
        == inj["loans"]["due_before_disbursement"]
    )
    assert (
        _check(pipeline, "loan_amount_not_positive")["failing_records"]
        == inj["loans"]["non_positive_amount"]
    )
    assert _check(pipeline, "duplicate_repayments")["failing_records"] == (
        inj["repayments"]["exact_duplicates"] + inj["repayments"]["reposted_duplicates"]
    )
    assert (
        _check(pipeline, "repayment_orphan_loan")["failing_records"]
        == inj["repayments"]["orphan_loan_id"]
    )
    assert (
        _check(pipeline, "repayment_amount_not_positive")["failing_records"]
        == inj["repayments"]["non_positive_amount"]
    )
    assert (
        _check(pipeline, "transaction_amount_missing")["failing_records"]
        == inj["transactions"]["null_amount"]
    )
    assert (
        _check(pipeline, "transaction_future_dated")["failing_records"]
        == inj["transactions"]["future_dated"]
    )
    assert (
        _check(pipeline, "duplicate_transactions")["failing_records"]
        == inj["transactions"]["exact_duplicates"]
    )


def test_injected_repayments_before_disbursement_are_quarantined(pipeline: PipelineRun) -> None:
    # Injected early-dated payments plus payments on loans whose dates were swapped.
    found = _check(pipeline, "repayment_before_disbursement")["failing_records"]
    assert found >= pipeline.manifest["injected_issues"]["repayments"]["before_disbursement"]


def test_income_unit_errors_are_flagged_not_deleted(pipeline: PipelineRun) -> None:
    flagged = pipeline.con.execute(
        "SELECT count(*) FILTER (WHERE is_income_outlier), "
        "count(*) FILTER (WHERE is_income_outlier AND monthly_income_clean IS NULL "
        "AND monthly_income IS NOT NULL) FROM clean.borrowers"
    ).fetchone()
    assert flagged[0] == pipeline.manifest["injected_issues"]["borrowers"]["income_unit_errors"]
    assert flagged[1] == flagged[0]


def test_labels_are_standardised(pipeline: PipelineRun) -> None:
    con = pipeline.con
    assert {
        r[0] for r in con.execute("SELECT DISTINCT gender FROM clean.borrowers").fetchall()
    } <= {"female", "male", "undisclosed"}
    channels = {
        r[0]
        for r in con.execute("SELECT DISTINCT payment_channel FROM clean.repayments").fetchall()
    }
    assert channels == {"mobile_app", "ussd", "agent", "bank_transfer", "auto_debit"}
    statuses = {
        r[0] for r in con.execute("SELECT DISTINCT source_loan_status FROM clean.loans").fetchall()
    }
    assert statuses <= {"active", "overdue", "repaid", "written_off"}


def test_quarantine_cascades_to_child_records(pipeline: PipelineRun) -> None:
    orphans = pipeline.con.execute(
        "SELECT count(*) FROM clean.repayments r "
        "WHERE r.loan_id IN (SELECT loan_id FROM clean.rejected_loans)"
    ).fetchone()[0]
    assert orphans == 0


def test_report_is_rendered(pipeline: PipelineRun) -> None:
    text = (pipeline.settings.paths.reports / "data_quality_report.md").read_text()
    assert "Clean-layer invariants" in text and "Quarantined records" in text
