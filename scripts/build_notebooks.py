"""Build and execute the storytelling notebooks in notebooks/.

Notebooks only *narrate*: every number comes from the warehouse or from functions in
the ``credit_risk`` package, so notebooks cannot drift from the pipeline. Run after
``make all``; they are executed in place so outputs are committed for readers.
"""

from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"

SETUP = """\
import duckdb
import pandas as pd

from credit_risk.config import get_settings

pd.set_option("display.width", 140)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:,.4f}".format)
settings = get_settings()
con = duckdb.connect(str(settings.paths.warehouse), read_only=True)
q = lambda sql: con.execute(sql).df()  # noqa: E731
"""


def sql_file(path: str) -> str:
    return f'q((settings.paths.sql / "{path}").read_text())'


NOTEBOOKS: dict[str, list[tuple[str, str]]] = {
    "01_portfolio_exploration.ipynb": [
        (
            "md",
            "# 01 - Portfolio exploration\n\nSynthetic digital-lending portfolio, "
            "snapshot 2025-12-31. This notebook reads the SQL marts built by `make etl`; "
            "definitions are in `docs/metrics.md`.",
        ),
        ("code", SETUP),
        ("md", "## Portfolio shape\nOne row per loan in `marts.loan_repayment_summary`."),
        (
            "code",
            "loans = q('SELECT * FROM marts.loan_repayment_summary')\n"
            "loans.groupby('product_type').agg(loans=('loan_id', 'size'), "
            "avg_amount=('loan_amount', 'mean'), avg_term=('term_days', 'mean'))",
        ),
        (
            "md",
            "## Headline KPIs recomputed in Python\nThe same functions the dashboard uses; "
            "`tests/test_marts.py` asserts they match the SQL marts.",
        ),
        (
            "code",
            "from credit_risk.analytics import metrics as lm\n"
            "pd.Series(lm.portfolio_kpis(loans, settings.risk, "
            "pd.Timestamp(settings.generator.snapshot_date)))",
        ),
        ("md", "## PAR trend (SQL)"),
        ("code", sql_file("analytics/03_par_by_month.sql") + ".tail(12)"),
        ("md", "## DPD distribution of the on-book portfolio"),
        ("code", sql_file("analytics/04_dpd_distribution.sql")),
        (
            "md",
            "**Reading.** Performing short-tenor loans leave the book within weeks, so the "
            "on-book stock over-represents delinquent loans. Flow measures - the 30+ DPD rate "
            "of each vintage - are the better gauge of new-business quality:",
        ),
        ("code", sql_file("analytics/12_cohort_vintage_analysis.sql")),
        ("md", "## Roll rates\nRow = bucket at month-end M, column = state at M+1."),
        ("code", sql_file("analytics/11_roll_rate_matrix.sql")),
    ],
    "02_risk_segmentation.ipynb": [
        (
            "md",
            "# 02 - Borrower risk segmentation\n\nRule-based segments "
            "(`sql/marts/07_borrower_mart.sql`) versus K-Means on behavioural features "
            "(`credit_risk.segmentation.clustering`).",
        ),
        ("code", SETUP),
        ("code", "bm = q('SELECT * FROM marts.borrower_mart')\nbm['risk_segment'].value_counts()"),
        ("md", "## Segment outcomes\nDescriptive only: segments are assigned with hindsight."),
        ("code", sql_file("analytics/08_default_rate_by_segment.sql")),
        ("md", "## Rules vs clusters"),
        (
            "code",
            "clusters = q('SELECT * FROM marts.borrower_clusters')\n"
            "pd.crosstab(clusters['risk_segment'], clusters['cluster'])",
        ),
        (
            "code",
            "from sklearn.metrics import adjusted_rand_score\n"
            "adjusted_rand_score(clusters['risk_segment'], clusters['cluster'])",
        ),
        (
            "md",
            "**Reading.** Moderate agreement: K-Means separates borrowers by engagement "
            "(tenure, frequency) as well as delinquency, whereas the rules are driven by "
            "severity and recency thresholds. The full discussion is in "
            "`reports/segmentation_report.md`.",
        ),
    ],
    "03_feature_engineering.ipynb": [
        (
            "md",
            "# 03 - Point-in-time feature engineering\n\nFeatures are built in SQL "
            "(`sql/features/01_loan_features.sql`) as of the day before each decision.",
        ),
        ("code", SETUP),
        (
            "code",
            "from credit_risk.features.registry import FEATURES\n"
            "pd.DataFrame([(f.name, f.group.value, f.kind.value, f.description) "
            "for f in FEATURES], columns=['feature', 'group', 'kind', 'description'])"
            ".groupby('group').size()",
        ),
        ("md", "## One repeat borrower's history, as the model saw it"),
        (
            "code",
            'loan = q("SELECT * FROM features.loan_features WHERE number_of_loans >= 3 '
            'ORDER BY loan_id LIMIT 1")\nloan.T.head(40)',
        ),
        (
            "md",
            "## Univariate signal on the training window\nA single feature with AUC > 0.85 "
            "would be flagged as potential leakage.",
        ),
        (
            "code",
            "from credit_risk.features.leakage import univariate_auc_screen\n"
            'train = q("SELECT * FROM features.model_dataset WHERE sample_split = '
            "'train'\")\nunivariate_auc_screen(train).head(12)",
        ),
        (
            "md",
            "## Leakage audit\nThe truncation test rebuilds features from data that "
            "existed before each cut-off and requires identical values "
            "(`reports/leakage_audit.md`).",
        ),
        (
            "code",
            "from credit_risk.features.leakage import default_cutoffs, truncation_test\n"
            "[(r.cutoff, r.loans_compared, r.mismatched_cells, r.passed) "
            "for r in truncation_test(con, settings, default_cutoffs(con))]",
        ),
    ],
    "04_model_validation.ipynb": [
        (
            "md",
            "# 04 - Model validation\n\nOut-of-time validation of the champion PD model. "
            "Reads `artifacts/evaluation/*.json` produced by `make evaluate`.",
        ),
        (
            "code",
            SETUP + "import json\n"
            "ev = json.loads((settings.paths.evaluation / 'evaluation.json').read_text())",
        ),
        ("code", "print(ev['metadata']['selection_rationale'])\npd.DataFrame(ev['diagnostics'])"),
        ("md", "## Predicted risk vs actual outcomes (test window)"),
        (
            "code",
            "pd.DataFrame(ev['deciles_test'])[['risk_decile', 'population', "
            "'avg_predicted_pd', 'actual_default_rate', 'lift', 'cum_bads_captured_pct']]",
        ),
        ("md", "## Calibration drift over time"),
        ("code", sql_file("analytics/17_actual_vs_predicted_by_month.sql") + ".tail(12)"),
        (
            "md",
            "## Why the latest months under-predict\nFeature PSI is stable while the "
            "realised bad rate rises: the late-2025 stress raised the *level* of risk "
            "without moving the inputs much. That calls for recalibration on matured "
            "recent outcomes, not a rebuild.",
        ),
        (
            "code",
            "mon = json.loads((settings.paths.evaluation / 'monitoring.json').read_text())\n"
            "pd.DataFrame(mon['performance'])",
        ),
        ("md", "## Responsible-modeling review"),
        (
            "code",
            "pd.DataFrame(ev['fairness_test'])[['attribute', 'group', 'n', 'bad_rate', "
            "'mean_pd', 'calibration_ratio', 'roc_auc', 'flagged_rate']]",
        ),
    ],
}


def build() -> None:
    OUT.mkdir(exist_ok=True)
    for name, cells in NOTEBOOKS.items():
        nb = new_notebook(
            cells=[
                new_markdown_cell(src) if kind == "md" else new_code_cell(src)
                for kind, src in cells
            ]
        )
        nb.metadata["kernelspec"] = {
            "name": "python3",
            "display_name": "Python 3",
            "language": "python",
        }
        NotebookClient(
            nb, timeout=600, kernel_name="python3", resources={"metadata": {"path": str(OUT)}}
        ).execute()
        nbformat.write(nb, OUT / name)
        print(f"executed {name}")


if __name__ == "__main__":
    build()
