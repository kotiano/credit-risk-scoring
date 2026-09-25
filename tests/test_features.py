"""Feature engineering: registry contract, point-in-time correctness, leakage audit."""

from __future__ import annotations

import shutil
from dataclasses import replace
from pathlib import Path

import pandas as pd
from conftest import PipelineRun

from credit_risk.config import Paths
from credit_risk.features.leakage import default_cutoffs, static_sql_scan, truncation_test
from credit_risk.features.registry import (
    IDENTIFIERS,
    REJECTED_CANDIDATES,
    categorical_features,
    numeric_features,
)


def test_feature_table_contains_every_registered_feature(pipeline: PipelineRun) -> None:
    cols = set(pipeline.con.execute("SELECT * FROM features.loan_features LIMIT 0").df().columns)
    assert set(numeric_features() + categorical_features()) <= cols
    assert set(IDENTIFIERS) <= cols


def test_sensitive_attributes_are_not_features(pipeline: PipelineRun) -> None:
    sensitive = set(pipeline.settings.modeling.sensitive_attributes)
    cols = set(pipeline.con.execute("SELECT * FROM features.loan_features LIMIT 0").df().columns)
    assert not sensitive & cols
    assert not sensitive & set(numeric_features() + categorical_features())
    assert {c.name for c in REJECTED_CANDIDATES} >= sensitive


def test_one_feature_row_per_loan(pipeline: PipelineRun) -> None:
    n, distinct, loans = pipeline.con.execute(
        "SELECT count(*), count(DISTINCT loan_id), (SELECT count(*) FROM clean.loans) "
        "FROM features.loan_features"
    ).fetchone()
    assert n == distinct == loans


def test_first_loans_have_no_history(pipeline: PipelineRun) -> None:
    df = pipeline.con.execute(
        "SELECT f.* FROM features.loan_features f JOIN clean.loans l USING (loan_id) "
        "WHERE l.borrower_loan_sequence = 1"
    ).df()
    assert (df["number_of_loans"] == 0).all()
    assert df["max_dpd"].eq(0).all() and df["repayment_rate"].isna().all()


def test_hand_computed_history_for_one_repeat_borrower(pipeline: PipelineRun) -> None:
    con = pipeline.con
    loan = (
        con.execute(
            "SELECT loan_id, borrower_id, decision_date FROM features.loan_features "
            "WHERE number_of_loans >= 2 ORDER BY loan_id LIMIT 1"
        )
        .df()
        .iloc[0]
    )
    prior = con.execute(
        "SELECT * FROM clean.loans WHERE borrower_id = ? AND disbursement_date < ?",
        [loan["borrower_id"], loan["decision_date"]],
    ).df()
    feats = (
        con.execute("SELECT * FROM features.loan_features WHERE loan_id = ?", [loan["loan_id"]])
        .df()
        .iloc[0]
    )
    assert feats["number_of_loans"] == len(prior)
    assert feats["total_disbursed"] == prior["loan_amount"].sum()
    assert feats["largest_loan"] == prior["loan_amount"].max()


def test_static_scan_finds_no_forbidden_references(pipeline: PipelineRun) -> None:
    assert static_sql_scan(pipeline.settings.paths.sql / "features" / "01_loan_features.sql") == []


def test_truncation_test_passes(pipeline: PipelineRun) -> None:
    results = truncation_test(pipeline.con, pipeline.settings, default_cutoffs(pipeline.con))
    assert results and all(r.passed for r in results)


def test_truncation_test_catches_a_deliberately_leaky_feature(
    pipeline: PipelineRun,
    tmp_path: Path,
) -> None:
    """Negative control: inject a feature that reads the prior loans' FINAL outcome."""
    sql_dir = tmp_path / "sql"
    shutil.copytree(pipeline.settings.paths.sql, sql_dir)
    feature_sql = sql_dir / "features" / "01_loan_features.sql"
    text = feature_sql.read_text()
    text = text.replace(
        "        max(pm.max_dpd_asof)                                        AS max_dpd,",
        "        max(s2.max_dpd_to_date)                                     AS max_dpd,",
    ).replace(
        "    FROM prior_measured AS pm\n    CROSS JOIN params AS prm",
        "    FROM prior_measured AS pm\n    JOIN marts.loan_repayment_summary AS s2\n"
        "        ON s2.loan_id = pm.prior_loan_id\n    CROSS JOIN params AS prm",
    )
    assert "s2.max_dpd_to_date" in text and "JOIN marts.loan_repayment_summary AS s2" in text
    feature_sql.write_text(text)
    leaky = replace(pipeline.settings, paths=replace(pipeline.settings.paths, sql=sql_dir))
    # Production features rebuilt with the leaky SQL, then compared with truncated rebuilds.
    from credit_risk.db import run_sql_file

    run_sql_file(pipeline.con, feature_sql)
    try:
        # Decision dates where a prior loan was still open, so its final DPD differs
        # from its DPD at the decision date.
        cutoffs = [
            d.date()
            for d in pipeline.con.execute(
                "SELECT DISTINCT decision_date FROM features.loan_features "
                "WHERE active_loans > 0 ORDER BY 1 LIMIT 15"
            ).df()["decision_date"]
        ]
        results = truncation_test(pipeline.con, leaky, cutoffs)
        assert any(not r.passed for r in results)
        assert any("max_dpd" in r.mismatched_columns for r in results)
        assert "max_dpd_to_date" in static_sql_scan(feature_sql)
    finally:
        run_sql_file(
            pipeline.con, pipeline.settings.paths.sql / "features" / "01_loan_features.sql"
        )


def test_model_dataset_splits_are_chronological(pipeline: PipelineRun) -> None:
    df = pipeline.con.execute("SELECT * FROM features.model_dataset").df()
    labelled = df[df["sample_split"] != "unobserved"]
    assert labelled["target"].notna().all()
    assert df.loc[df["sample_split"] == "unobserved", "target"].isna().all()
    order = ["train", "validation", "test", "recent"]
    ranges = labelled.groupby("sample_split")["decision_date"].agg(["min", "max"]).loc[order]
    assert (ranges["max"].iloc[:-1].to_numpy() < ranges["min"].iloc[1:].to_numpy()).all()


def test_paths_default_is_untouched() -> None:
    assert Paths().sql.name == "sql"
    assert isinstance(pd.Timestamp.now(), pd.Timestamp)
