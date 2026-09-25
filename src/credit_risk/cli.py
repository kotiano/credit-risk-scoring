"""Command-line entry point: ``python -m credit_risk.cli <command>``.

Each command is one pipeline stage; ``all`` runs them in order. Stages communicate only
through the DuckDB warehouse and ``artifacts/`` so any stage can be re-run on its own.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections.abc import Callable

from credit_risk.config import Settings, get_settings
from credit_risk.db import connect, prepare_session
from credit_risk.logging_utils import configure_logging, get_logger, log_event

logger = get_logger("credit_risk.cli")


def cmd_generate(settings: Settings) -> None:
    from credit_risk.ingestion.synthetic import generate_portfolio, write_raw_extracts

    tables, manifest = generate_portfolio(settings.generator)
    write_raw_extracts(tables, manifest, settings.paths.raw_data)


def cmd_validate(settings: Settings) -> None:
    from credit_risk.ingestion.load_raw import load_raw_extracts
    from credit_risk.transformations.pipeline import run_layers
    from credit_risk.validation.report import render_data_quality_report
    from credit_risk.validation.runner import run_data_quality

    with connect(settings.paths.warehouse) as con:
        load_raw_extracts(con, settings.paths.raw_data)
        run_layers(con, settings, ("staging", "clean"))
        payload = run_data_quality(con, settings.paths.artifacts)
    render_data_quality_report(payload, settings.paths.reports / "data_quality_report.md")


def cmd_etl(settings: Settings) -> None:
    from credit_risk.transformations.pipeline import run_layers

    with connect(settings.paths.warehouse) as con:
        run_layers(con, settings, ("marts",))


def cmd_segment(settings: Settings) -> None:
    from credit_risk.segmentation.clustering import run_segmentation

    with connect(settings.paths.warehouse) as con:
        prepare_session(con, settings)
        run_segmentation(con, settings)


def cmd_features(settings: Settings) -> None:
    from credit_risk.features.audit import run_leakage_audit
    from credit_risk.transformations.pipeline import run_feature_sql

    with connect(settings.paths.warehouse) as con:
        run_feature_sql(con, settings)
        run_leakage_audit(con, settings)


def cmd_train(settings: Settings) -> None:
    from credit_risk.modeling.train import persist_training, train_models

    with connect(settings.paths.warehouse) as con:
        prepare_session(con, settings)
        persist_training(train_models(con, settings), con, settings)


def cmd_evaluate(settings: Settings) -> None:
    from credit_risk.modeling.evaluate import evaluate_models

    with connect(settings.paths.warehouse) as con:
        prepare_session(con, settings)
        evaluate_models(con, settings)


def cmd_monitor(settings: Settings) -> None:
    from credit_risk.monitoring.monitor import run_monitoring

    with connect(settings.paths.warehouse) as con:
        prepare_session(con, settings)
        run_monitoring(con, settings)


def cmd_analytics(settings: Settings) -> None:
    from credit_risk.analytics.sql_reports import run_analytics

    with connect(settings.paths.warehouse) as con:
        prepare_session(con, settings)
        run_analytics(con, settings)


COMMANDS: dict[str, Callable[[Settings], None]] = {
    "generate-data": cmd_generate,
    "validate": cmd_validate,
    "etl": cmd_etl,
    "segment": cmd_segment,
    "features": cmd_features,
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "monitor": cmd_monitor,
    "analytics": cmd_analytics,
}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="credit-risk", description=__doc__)
    parser.add_argument("command", choices=[*COMMANDS, "all"])
    args = parser.parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    stages = list(COMMANDS) if args.command == "all" else [args.command]
    for stage in stages:
        started = time.perf_counter()
        COMMANDS[stage](settings)
        log_event(
            logger, "stage complete", stage=stage, seconds=round(time.perf_counter() - started, 1)
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
