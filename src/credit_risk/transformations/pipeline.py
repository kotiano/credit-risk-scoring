"""Orchestrates the SQL transformation layers (staging -> clean -> marts -> features)."""

from __future__ import annotations

import duckdb

from credit_risk.config import Settings
from credit_risk.db import prepare_session, run_sql_layer
from credit_risk.logging_utils import get_logger, log_event

logger = get_logger(__name__)

ETL_LAYERS = ("staging", "clean", "marts")


def run_layers(con: duckdb.DuckDBPyConnection, settings: Settings, layers: tuple[str, ...]) -> None:
    prepare_session(con, settings)
    for layer in layers:
        files = run_sql_layer(con, settings.paths.sql / layer)
        log_event(logger, "layer complete", layer=layer, files=len(files))


def run_etl(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    run_layers(con, settings, ETL_LAYERS)


def run_feature_sql(con: duckdb.DuckDBPyConnection, settings: Settings) -> None:
    run_layers(con, settings, ("features",))
