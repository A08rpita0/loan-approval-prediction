"""Train, compare and evaluate loan approval models.

Usage::

    python -m loan_approval.train
    python -m loan_approval.train --sample 800 --out-dir some/dir   # quick run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

from loan_approval import config
from loan_approval.data import FEATURE_SETS, build_features, load_dataset

logger = logging.getLogger(__name__)

SELECTION_METRIC = "roc_auc"
# Candidates within this CV ROC-AUC margin of the best are treated as tied;
# the simplest (earliest-listed) one wins.
TIE_TOLERANCE = 0.001


@dataclass(frozen=True)
class Candidate:
    """A model + feature-set combination evaluated with cross-validation."""

    model_name: str
    feature_set: str

    @property
    def key(self) -> str:
        return f"{self.model_name}/{self.feature_set}"


def make_estimator(model_name: str) -> ClassifierMixin:
    """Return an unfitted classifier by name."""
    if model_name == "logistic_regression":
        return LogisticRegression(max_iter=1000)
    if model_name == "hist_gradient_boosting":
        # Domain constraint: a higher CIBIL score must never lower approval odds.
        # Without it the boosted trees were non-monotonic in CIBIL on every probed row.
        return HistGradientBoostingClassifier(
            monotonic_cst={"cibil_score": 1}, random_state=config.RANDOM_SEED
        )
    raise ValueError(f"Unknown model {model_name!r}")


def build_pipeline(model_name: str, feature_set: str) -> Pipeline:
    """Build the full pipeline: raw inputs -> features -> encoding/scaling -> model.

    Categorical columns are passed as strings ("Graduate", "Yes") and encoded
    inside the pipeline, so callers can never apply a mismatched manual encoding.
    """
    categorical = list(config.ALLOWED_CATEGORIES)
    preprocess = ColumnTransformer(
        [
            (
                "cat",
                OneHotEncoder(
                    categories=[config.ALLOWED_CATEGORIES[c] for c in categorical],
                    drop="first",
                    handle_unknown="error",
                    sparse_output=False,
                ),
                categorical,
            ),
            ("num", StandardScaler(), make_column_selector(dtype_include=np.number)),
        ],
        verbose_feature_names_out=False,
    ).set_output(transform="pandas")  # keeps feature names (needed for monotonic_cst)
    return Pipeline(
        [
            ("features", FunctionTransformer(build_features, kw_args={"feature_set": feature_set})),
            ("preprocess", preprocess),
            ("model", make_estimator(model_name)),
        ]
    )


def candidates() -> list[Candidate]:
    """All candidates, ordered simplest first (used for tie-breaking)."""
    return [
        Candidate(m, fs)
        for m in ("logistic_regression", "hist_gradient_boosting")
        for fs in FEATURE_SETS
    ]


def file_sha256(path: Path) -> str:
    """SHA-256 of a file, tying an artifact to the exact data it was trained on."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(y_true: pd.Series, proba: np.ndarray, threshold: float) -> dict[str, Any]:
    """Compute held-out classification metrics."""
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    return {
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "classification_report": classification_report(
            y_true, pred, target_names=["Rejected", "Approved"], output_dict=True, zero_division=0
        ),
    }


def run(
    data_path: Path = config.DATA_PATH,
    out_dir: Path = config.MODELS_DIR,
    sample: int | None = None,
) -> dict[str, Any]:
    """Run the full training workflow and write artifact, metrics and model card.

    Returns:
        The metrics dictionary written to ``metrics.json``.
    """
    X, y = load_dataset(data_path)
    if sample is not None and sample < len(X):
        X, _, y, _ = train_test_split(
            X, y, train_size=sample, stratify=y, random_state=config.RANDOM_SEED
        )
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE, stratify=y, random_state=config.RANDOM_SEED
    )
    logger.info("Train rows: %d, test rows: %d", len(X_train), len(X_test))

    cv = StratifiedKFold(n_splits=config.CV_FOLDS, shuffle=True, random_state=config.RANDOM_SEED)
    scoring = ["roc_auc", "f1", "accuracy", "average_precision"]
    cv_results: dict[str, dict[str, float]] = {}
    for cand in candidates():
        scores = cross_validate(
            build_pipeline(cand.model_name, cand.feature_set),
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
        )
        cv_results[cand.key] = {
            f"{m}_{stat}": float(getattr(np, stat)(scores[f"test_{m}"]))
            for m in scoring
            for stat in ("mean", "std")
        }
        logger.info(
            "CV %-45s ROC-AUC %.4f +/- %.4f",
            cand.key,
            cv_results[cand.key]["roc_auc_mean"],
            cv_results[cand.key]["roc_auc_std"],
        )

    best = max(r[f"{SELECTION_METRIC}_mean"] for r in cv_results.values())
    chosen = next(
        c
        for c in candidates()
        if cv_results[c.key][f"{SELECTION_METRIC}_mean"] >= best - TIE_TOLERANCE
    )
    logger.info("Selected %s", chosen.key)

    pipeline = build_pipeline(chosen.model_name, chosen.feature_set).fit(X_train, y_train)
    proba = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, proba, config.DECISION_THRESHOLD)

    perm = permutation_importance(
        pipeline, X_test, y_test, scoring="roc_auc", n_repeats=10, random_state=config.RANDOM_SEED
    )
    importances = {
        col: {"mean": float(m), "std": float(s)}
        for col, m, s in sorted(
            zip(config.INPUT_COLUMNS, perm.importances_mean, perm.importances_std, strict=True),
            key=lambda t: -t[1],
        )
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / config.MODEL_PATH.name
    joblib.dump(pipeline, model_path)

    metrics: dict[str, Any] = {
        "chosen_model": chosen.model_name,
        "feature_set": chosen.feature_set,
        "selection": {
            "metric": f"cv_{SELECTION_METRIC}",
            "tie_tolerance": TIE_TOLERANCE,
            "rule": "simplest candidate within tie_tolerance of the best mean CV score",
        },
        "decision_threshold": config.DECISION_THRESHOLD,
        "input_columns": config.INPUT_COLUMNS,
        "model_features": [
            str(f) for f in pipeline.named_steps["preprocess"].get_feature_names_out()
        ],
        "cv": {"folds": config.CV_FOLDS, "results": cv_results},
        "test": test_metrics,
        "permutation_importance_test_roc_auc": importances,
        "data": {
            "path": data_path.name,
            "sha256": file_sha256(data_path),
            "n_rows": len(X),
            "n_train": len(X_train),
            "n_test": len(X_test),
            "approved_rate": float(y.mean()),
            "sampled": sample is not None,
        },
        "random_seed": config.RANDOM_SEED,
        "sklearn_version": sklearn.__version__,
        "trained_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }
    (out_dir / config.METRICS_PATH.name).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out_dir / config.MODEL_CARD_PATH.name).write_text(render_model_card(metrics), encoding="utf-8")
    logger.info("Wrote artifact to %s", model_path)
    return metrics


def render_model_card(m: dict[str, Any]) -> str:
    """Render a markdown model card from the metrics dictionary."""
    t, cm, d = m["test"], m["test"]["confusion_matrix"], m["data"]
    cv_rows = "\n".join(
        f"| {k} | {v['roc_auc_mean']:.4f} ± {v['roc_auc_std']:.4f} "
        f"| {v['f1_mean']:.4f} ± {v['f1_std']:.4f} | {v['accuracy_mean']:.4f} |"
        for k, v in m["cv"]["results"].items()
    )
    imp_rows = "\n".join(
        f"| {k} | {v['mean']:.4f} ± {v['std']:.4f} |"
        for k, v in m["permutation_importance_test_roc_auc"].items()
    )
    return f"""# Model Card: Loan Approval Classifier

_Generated automatically by `python -m loan_approval.train` on {m["trained_at"]}._

## Model
- **Type:** `{m["chosen_model"]}` with feature set `{m["feature_set"]}` (scikit-learn {m["sklearn_version"]})
- **Artifact:** one sklearn `Pipeline` (feature engineering, one-hot encoding, scaling, model)
- **Decision threshold:** {m["decision_threshold"]} on predicted approval probability
- **Selection:** {m["selection"]["rule"]} ({m["selection"]["metric"]}, tolerance {m["selection"]["tie_tolerance"]})

## Data
- `{d["path"]}`, {d["n_rows"]} rows (sha256 `{d["sha256"][:12]}...`), approval rate {d["approved_rate"]:.3f}
- Stratified split: {d["n_train"]} train / {d["n_test"]} test, seed {m["random_seed"]}

## Cross-validation ({m["cv"]["folds"]}-fold stratified, train split only)
| Candidate | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
{cv_rows}

## Held-out test metrics (n={d["n_test"]})
| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| {t["accuracy"]:.4f} | {t["precision"]:.4f} | {t["recall"]:.4f} | {t["f1"]:.4f} | {t["roc_auc"]:.4f} | {t["pr_auc"]:.4f} |

Confusion matrix (rows = actual, columns = predicted; positive class = Approved):

| | Pred. Rejected | Pred. Approved |
|---|---|---|
| **Rejected** | {cm["tn"]} | {cm["fp"]} |
| **Approved** | {cm["fn"]} | {cm["tp"]} |

## Permutation importance (drop in test ROC-AUC, raw input columns)
| Input | Importance |
|---|---|
{imp_rows}

## Intended use and limitations
Educational demo trained on a public Kaggle dataset. Not validated for real
credit decisions. The data is dominated by one feature (CIBIL score), which is
why scores are high; they do not transfer to real lending portfolios.
No fairness audit has been performed.
"""


def main() -> None:
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--data", type=Path, default=config.DATA_PATH)
    parser.add_argument("--out-dir", type=Path, default=config.MODELS_DIR)
    parser.add_argument("--sample", type=int, default=None, help="Train on a stratified subsample")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    # OpenMP oversubscription made HistGradientBoosting ~25x slower on small data
    # on some Windows machines; single-threaded is fast enough for 4k rows.
    with threadpool_limits(limits=1):
        metrics = run(args.data, args.out_dir, args.sample)
    t = metrics["test"]
    logger.info("Test ROC-AUC %.4f | F1 %.4f | accuracy %.4f", t["roc_auc"], t["f1"], t["accuracy"])


if __name__ == "__main__":
    main()
