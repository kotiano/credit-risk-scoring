"""Smoke-test the Streamlit dashboard against the test pipeline outputs."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import pytest
from conftest import PipelineRun
from streamlit.testing.v1 import AppTest

APP = Path(__file__).resolve().parents[1] / "dashboard" / "app.py"


def test_dashboard_renders_all_tabs_without_errors(
    pipeline: PipelineRun,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # DuckDB allows one configuration per file per process; the fixture holds a
    # read-write connection, so the read-only dashboard uses a checkpointed copy.
    pipeline.con.execute("CHECKPOINT")
    copy = tmp_path / "warehouse.duckdb"
    shutil.copy(pipeline.settings.paths.warehouse, copy)
    monkeypatch.setenv("CREDIT_RISK_DB_PATH", str(copy))
    monkeypatch.setenv("CREDIT_RISK_ARTIFACTS_DIR", str(pipeline.settings.paths.artifacts))
    for mod in ("loaders", "theme"):
        os.sys.modules.pop(mod, None)
    at = AppTest.from_file(str(APP), default_timeout=120).run()
    assert not at.exception, [e.value for e in at.exception]
    assert len(at.tabs) == 8
    # Filtering to one product recomputes metrics without errors.
    at.sidebar.multiselect[0].set_value(["micro_loan"]).run()
    assert not at.exception
    assert "micro_loan" in at.sidebar.multiselect[0].value
