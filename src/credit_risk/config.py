"""Centralised project configuration.

Every business rule that changes a metric (DPD buckets, PAR thresholds, default and
write-off definitions, split dates, risk bands) lives here so that SQL, Python and the
dashboard read the same values. SQL receives these values as DuckDB session variables
(see :func:`credit_risk.db.sql_variables`), so the definitions cannot silently diverge.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT / ".env")


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


@dataclass(frozen=True)
class Paths:
    """Filesystem layout. All generated outputs are reproducible and git-ignored,
    except the Markdown reports, which are committed so reviewers can read results."""

    root: Path = PROJECT_ROOT
    sql: Path = PROJECT_ROOT / "sql"
    raw_data: Path = PROJECT_ROOT / "data" / "raw"
    warehouse: Path = field(
        default_factory=lambda: (
            PROJECT_ROOT / os.getenv("CREDIT_RISK_DB_PATH", "data/warehouse/credit_risk.duckdb")
        )
    )
    artifacts: Path = field(
        default_factory=lambda: PROJECT_ROOT / os.getenv("CREDIT_RISK_ARTIFACTS_DIR", "artifacts")
    )
    reports: Path = PROJECT_ROOT / "reports"

    @property
    def models(self) -> Path:
        return self.artifacts / "models"

    @property
    def evaluation(self) -> Path:
        return self.artifacts / "evaluation"


@dataclass(frozen=True)
class GeneratorConfig:
    """Synthetic portfolio parameters (see docs/assumptions.md)."""

    n_borrowers: int = field(default_factory=lambda: _env_int("CREDIT_RISK_N_BORROWERS", 5000))
    seed: int = field(default_factory=lambda: _env_int("CREDIT_RISK_SEED", 42))
    portfolio_start: date = date(2024, 1, 1)
    snapshot_date: date = date(2025, 12, 31)
    # Macro-economic stress episode used to create realistic drift in late 2025.
    stress_start: date = date(2025, 8, 1)
    inject_quality_issues: bool = True


@dataclass(frozen=True)
class RiskDefinitions:
    """Credit-risk metric definitions shared by SQL and Python.

    DPD buckets are lower-inclusive and aligned to the PAR thresholds, so PAR_n equals
    the outstanding principal in every bucket at or above ``n`` (docs/metrics.md).
    """

    # A balance at or below this amount is treated as settled (rounding / small residue).
    settlement_tolerance: float = 1.0
    par_thresholds: tuple[int, ...] = (1, 7, 30, 60, 90)
    # (lower bound inclusive, label). Upper bound is the next lower bound - 1.
    dpd_buckets: tuple[tuple[int, str], ...] = (
        (0, "current"),
        (1, "1-6"),
        (7, "7-29"),
        (30, "30-59"),
        (60, "60-89"),
        (90, "90+"),
    )
    default_dpd: int = 90  # Basel-style 90 days past due
    write_off_dpd: int = 180  # loans are charged off and leave the PAR denominator
    target_dpd: int = 30  # model target: loan reaches 30 DPD
    early_repayment_min_days: int = 3  # settled >= 3 days before the due date

    def __post_init__(self) -> None:
        # Guard against the bucket table and PAR thresholds drifting apart.
        lower_bounds = [lb for lb, _ in self.dpd_buckets]
        if lower_bounds != sorted(lower_bounds) or lower_bounds[0] != 0:
            raise ValueError("DPD buckets must start at 0 and be sorted")
        if not set(self.par_thresholds).issubset(lower_bounds):
            raise ValueError("Every PAR threshold must be a DPD bucket lower bound")

    @property
    def bucket_labels(self) -> list[str]:
        return [label for _, label in self.dpd_buckets]


@dataclass(frozen=True)
class ModelingConfig:
    """Out-of-time split and scoring conventions (docs/modeling.md)."""

    # Loans are assigned to a split by disbursement date (decision date).
    train_end: date = date(2025, 3, 31)
    validation_end: date = date(2025, 6, 30)
    test_end: date = date(2025, 9, 30)
    random_state: int = 42
    # Parsimony rule: keep the interpretable baseline unless a challenger beats it
    # by more than this validation ROC-AUC margin.
    parsimony_auc_margin: float = 0.01
    # Points-to-double-odds scaling: score = offset + factor * ln(odds_good_bad)
    score_base: float = 600.0
    score_base_odds: float = 50.0
    score_pdo: float = 20.0
    # PD upper bounds (exclusive) for analytical risk bands; last band is open-ended.
    risk_bands: tuple[tuple[float, str], ...] = (
        (0.04, "A - Very Low"),
        (0.08, "B - Low"),
        (0.15, "C - Moderate"),
        (0.30, "D - Elevated"),
        (1.01, "E - High"),
    )
    # Attributes excluded from model features for responsible-modeling reasons.
    sensitive_attributes: tuple[str, ...] = ("gender", "age", "region")


@dataclass(frozen=True)
class MonitoringConfig:
    psi_bins: int = 10
    psi_moderate: float = 0.10
    psi_significant: float = 0.25


@dataclass(frozen=True)
class Settings:
    paths: Paths = field(default_factory=Paths)
    generator: GeneratorConfig = field(default_factory=GeneratorConfig)
    risk: RiskDefinitions = field(default_factory=RiskDefinitions)
    modeling: ModelingConfig = field(default_factory=ModelingConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    log_level: str = field(default_factory=lambda: os.getenv("CREDIT_RISK_LOG_LEVEL", "INFO"))
    # Parallel floating-point aggregation is not bit-for-bit deterministic; one thread
    # makes reruns reproduce the committed reports exactly (cost: a few seconds).
    duckdb_threads: int = field(default_factory=lambda: _env_int("CREDIT_RISK_DUCKDB_THREADS", 1))


def get_settings() -> Settings:
    return Settings()
