"""Model performance metrics for imbalanced credit-risk classification.

Why each metric matters (docs/validation.md):

* ROC-AUC / Gini (= 2*AUC - 1): rank-ordering power, threshold-free; the standard
  discrimination measure in credit scoring.
* KS: maximum separation between the cumulative score distributions of goods and bads.
* PR-AUC (average precision): discrimination focused on the minority (bad) class; its
  baseline equals the bad rate, so always read it relative to prevalence.
* Brier score / log loss: accuracy of the probabilities themselves (PD calibration).
* Calibration-in-the-large (mean PD / observed bad rate) and ECE: whether predicted
  PDs can be used as PDs (pricing, expected loss), not only for ranking.
* Precision, recall (sensitivity), specificity, F1 at a threshold: operating trade-offs
  once a cut-off is chosen. Accuracy is deliberately not used for selection.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class ThresholdMetrics:
    threshold: float
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    specificity: float
    f1: float
    flagged_rate: float  # share of applications at or above the threshold


def ks_statistic(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic = max |TPR - FPR| over thresholds."""
    fpr, tpr, _ = roc_curve(y_true, y_score)
    return float(np.max(tpr - fpr))


def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> float:
    """Population-weighted mean |observed - predicted| over equal-frequency bins."""
    table = calibration_table(y_true, y_prob, bins)
    weights = table["n"] / table["n"].sum()
    return float((weights * (table["observed_rate"] - table["mean_pd"]).abs()).sum())


def threshold_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> ThresholdMetrics:
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return ThresholdMetrics(
        float(threshold),
        int(tp),
        int(fp),
        int(tn),
        int(fn),
        float(precision),
        float(recall),
        float(specificity),
        float(f1),
        float(y_pred.mean()),
    )


def classification_report(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float
) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.clip(np.asarray(y_prob, dtype=float), 1e-6, 1 - 1e-6)
    auc = roc_auc_score(y_true, y_prob)
    at = threshold_metrics(y_true, y_prob, threshold)
    return {
        "n": len(y_true),
        "bad_rate": float(y_true.mean()),
        "mean_pd": float(y_prob.mean()),
        "roc_auc": float(auc),
        "gini": float(2 * auc - 1),
        "ks": ks_statistic(y_true, y_prob),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "brier": float(brier_score_loss(y_true, y_prob)),
        "log_loss": float(log_loss(y_true, y_prob, labels=[0, 1])),
        "calibration_ratio": float(y_prob.mean() / y_true.mean()) if y_true.mean() else np.nan,
        "ece": expected_calibration_error(y_true, y_prob),
        **{k: v for k, v in asdict(at).items() if k not in {"tp", "fp", "tn", "fn"}},
    }


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    """Equal-frequency bins of predicted PD vs observed default rate."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, float)})
    df["bin"] = pd.qcut(df["p"].rank(method="first"), bins, labels=False)
    out = df.groupby("bin").agg(
        n=("y", "size"), mean_pd=("p", "mean"), observed_rate=("y", "mean"), bads=("y", "sum")
    )
    return out.reset_index()


def decile_table(y_true: np.ndarray, y_prob: np.ndarray, deciles: int = 10) -> pd.DataFrame:
    """Risk deciles (1 = highest predicted PD) with lift and cumulative gains."""
    df = pd.DataFrame({"y": np.asarray(y_true).astype(int), "p": np.asarray(y_prob, float)})
    df["risk_decile"] = deciles - pd.qcut(df["p"].rank(method="first"), deciles, labels=False)
    out = (
        df.groupby("risk_decile")
        .agg(
            population=("y", "size"),
            min_pd=("p", "min"),
            max_pd=("p", "max"),
            avg_predicted_pd=("p", "mean"),
            actual_default_rate=("y", "mean"),
            bads=("y", "sum"),
        )
        .sort_index()
    )
    total_bads, overall_rate = out["bads"].sum(), df["y"].mean()
    out["lift"] = out["actual_default_rate"] / overall_rate
    out["cum_population_pct"] = out["population"].cumsum() / out["population"].sum()
    out["cum_bads_captured_pct"] = out["bads"].cumsum() / total_bads  # gains
    out["cum_lift"] = out["cum_bads_captured_pct"] / out["cum_population_pct"]
    return out.reset_index()


def threshold_table(
    y_true: np.ndarray, y_prob: np.ndarray, thresholds: list[float] | None = None
) -> pd.DataFrame:
    """Trade-offs if applications with PD >= threshold were declined (analytical only)."""
    y_true = np.asarray(y_true).astype(int)
    thresholds = thresholds or [0.05, 0.08, 0.10, 0.12, 0.15, 0.20, 0.25, 0.30, 0.40]
    rows = []
    for t in thresholds:
        m = threshold_metrics(y_true, y_prob, t)
        approved = y_prob < t
        rows.append(
            {
                **asdict(m),
                "approval_rate": float(approved.mean()),
                "bad_rate_approved": float(y_true[approved].mean()) if approved.any() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def best_f1_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
    f1 = np.where(precision + recall > 0, 2 * precision * recall / (precision + recall), 0)
    return float(thresholds[int(np.argmax(f1[:-1]))])


def curve_points(y_true: np.ndarray, y_prob: np.ndarray, max_points: int = 200) -> dict:
    """Down-sampled ROC and PR curve coordinates for plotting."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    precision, recall, _ = precision_recall_curve(y_true, y_prob)

    def sample(*arrays: np.ndarray) -> list[list[float]]:
        idx = np.unique(np.linspace(0, len(arrays[0]) - 1, max_points).astype(int))
        return [a[idx].round(5).tolist() for a in arrays]

    roc_x, roc_y = sample(fpr, tpr)
    pr_x, pr_y = sample(recall, precision)
    return {"roc": {"fpr": roc_x, "tpr": roc_y}, "pr": {"recall": pr_x, "precision": pr_y}}
