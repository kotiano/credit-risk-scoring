# Credit Risk Analytics & Model Validation - task runner
# Every stage is reproducible from a clean checkout: `make setup all test`.

PYTHON  ?= python3.12
VENV    ?= .venv
BIN     := $(VENV)/bin
CLI     := $(BIN)/python -m credit_risk.cli

.DEFAULT_GOAL := help
.PHONY: help setup generate-data validate etl segment features train evaluate monitor \
        analytics all dashboard notebooks test lint format typecheck check clean

help:  ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

setup:  ## Create the virtualenv and install the project with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

generate-data:  ## Generate the synthetic raw extracts (data/raw)
	$(CLI) generate-data

validate:  ## Load raw -> staging -> clean and run data-quality checks
	$(CLI) validate

etl:  ## Build marts (loan summary, DPD/PAR snapshots, borrower mart, cohorts)
	$(CLI) etl

segment:  ## Rule-based vs K-Means borrower segmentation report
	$(CLI) segment

features:  ## Build point-in-time features and run the leakage audit
	$(CLI) features

train:  ## Train candidate models out-of-time and select the champion
	$(CLI) train

evaluate:  ## Validation report: predicted vs actual, calibration, fairness
	$(CLI) evaluate

monitor:  ## PSI / drift / performance-over-time monitoring
	$(CLI) monitor

analytics:  ## Run the SQL analytics library (reports/sql_analytics.md)
	$(CLI) analytics

all:  ## Run the full pipeline end-to-end
	$(CLI) all

dashboard:  ## Launch the Streamlit dashboard (after `make all`)
	$(BIN)/streamlit run dashboard/app.py

notebooks:  ## Re-execute the storytelling notebooks in place
	$(BIN)/python scripts/build_notebooks.py
	$(BIN)/ruff format -q notebooks
	$(BIN)/ruff check --fix -q notebooks
	$(BIN)/ruff format -q notebooks

test:  ## Run the test suite
	$(BIN)/pytest

lint:  ## Ruff lint + format check
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:  ## Auto-format
	$(BIN)/ruff check --fix .
	$(BIN)/ruff format .

typecheck:  ## mypy on the package
	$(BIN)/mypy

check: lint typecheck test  ## Lint, type-check and test

clean:  ## Remove generated data and artifacts (reports are kept)
	rm -rf data/raw data/warehouse artifacts
