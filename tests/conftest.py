"""Shared fixtures.

``pipeline`` runs the complete pipeline once per test session on a small synthetic
portfolio in a temporary directory, so integration tests never touch the real
warehouse, artifacts or committed reports.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pytest

from credit_risk.config import GeneratorConfig, Paths, Settings
from credit_risk.db import connect, prepare_session


@dataclass
class PipelineRun:
    settings: Settings
    con: duckdb.DuckDBPyConnection
    manifest: dict
    dq: dict


def make_settings(tmp: Path, n_borrowers: int = 600, seed: int = 11) -> Settings:
    paths = Paths(
        raw_data=tmp / "raw",
        warehouse=tmp / "warehouse.duckdb",
        artifacts=tmp / "artifacts",
        reports=tmp / "reports",
    )
    return Settings(paths=paths, generator=GeneratorConfig(n_borrowers=n_borrowers, seed=seed))


@pytest.fixture(scope="session")
def pipeline(tmp_path_factory: pytest.TempPathFactory) -> Iterator[PipelineRun]:
    from credit_risk.analytics.sql_reports import run_analytics
    from credit_risk.features.audit import run_leakage_audit
    from credit_risk.ingestion.load_raw import load_raw_extracts
    from credit_risk.ingestion.synthetic import generate_portfolio, write_raw_extracts
    from credit_risk.modeling.evaluate import evaluate_models
    from credit_risk.modeling.train import persist_training, train_models
    from credit_risk.monitoring.monitor import run_monitoring
    from credit_risk.segmentation.clustering import run_segmentation
    from credit_risk.transformations.pipeline import run_feature_sql, run_layers
    from credit_risk.validation.report import render_data_quality_report
    from credit_risk.validation.runner import run_data_quality

    tmp = tmp_path_factory.mktemp("pipeline")
    settings = make_settings(tmp)
    settings.paths.reports.mkdir(parents=True)
    tables, manifest = generate_portfolio(settings.generator)
    write_raw_extracts(tables, manifest, settings.paths.raw_data)

    con = connect(settings.paths.warehouse)
    load_raw_extracts(con, settings.paths.raw_data)
    run_layers(con, settings, ("staging", "clean"))
    dq = run_data_quality(con, settings.paths.artifacts)
    render_data_quality_report(dq, settings.paths.reports / "data_quality_report.md")
    run_layers(con, settings, ("marts",))
    run_segmentation(con, settings)
    run_feature_sql(con, settings)
    run_leakage_audit(con, settings)
    persist_training(train_models(con, settings), con, settings)
    evaluate_models(con, settings)
    run_monitoring(con, settings)
    run_analytics(con, settings)
    prepare_session(con, settings)
    yield PipelineRun(
        settings, con, json.loads((settings.paths.raw_data / "_manifest.json").read_text()), dq
    )
    con.close()
