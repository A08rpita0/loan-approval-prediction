"""Inference: validated application input, cached model loading and explanations."""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from loan_approval import config

logger = logging.getLogger(__name__)


class ModelNotFoundError(FileNotFoundError):
    """Raised when the trained artifact is missing."""


@dataclass(frozen=True)
class LoanApplication:
    """A single loan application in raw (dataset) units.

    Categorical fields use the dataset's string labels; encoding happens
    inside the model pipeline. Monetary fields are in the dataset currency (INR).
    """

    no_of_dependents: int
    education: str  # "Graduate" | "Not Graduate"
    self_employed: str  # "No" | "Yes"
    income_annum: float
    loan_amount: float
    loan_term: int  # years
    cibil_score: int
    residential_assets_value: float = 0.0
    commercial_assets_value: float = 0.0
    luxury_assets_value: float = 0.0
    bank_asset_value: float = 0.0

    def __post_init__(self) -> None:
        errors = self.validation_errors()
        if errors:
            raise ValueError("; ".join(errors))

    def validation_errors(self) -> list[str]:
        """Return human-readable validation errors (empty if valid)."""
        errors: list[str] = []
        for name, allowed in config.ALLOWED_CATEGORIES.items():
            if getattr(self, name) not in allowed:
                errors.append(f"{name} must be one of {allowed}, got {getattr(self, name)!r}")
        lo, hi = config.CIBIL_RANGE
        if not lo <= self.cibil_score <= hi:
            errors.append(f"cibil_score must be between {lo} and {hi}")
        lo, hi = config.DEPENDENTS_RANGE
        if not lo <= self.no_of_dependents <= hi:
            errors.append(f"no_of_dependents must be between {lo} and {hi}")
        lo, hi = config.LOAN_TERM_RANGE
        if not lo <= self.loan_term <= hi:
            errors.append(f"loan_term must be between {lo} and {hi} years")
        if self.income_annum <= 0:
            errors.append("income_annum must be positive")
        if self.loan_amount <= 0:
            errors.append("loan_amount must be positive")
        for col in config.ASSET_COLUMNS:
            if getattr(self, col) < 0:
                errors.append(f"{col} cannot be negative")
        return errors

    def to_frame(self) -> pd.DataFrame:
        """One-row dataframe with columns in the order the pipeline expects."""
        return pd.DataFrame([asdict(self)])[config.INPUT_COLUMNS]


@dataclass
class PredictionResult:
    """Model output for one application."""

    approval_probability: float
    approved: bool
    threshold: float
    # (factor, value) pairs sorted by absolute impact.
    factors: list[tuple[str, float]] = field(default_factory=list)
    # "local" = per-application log-odds contributions; "global" = permutation importance.
    explanation_kind: str = "global"

    @property
    def label(self) -> str:
        return "Approved" if self.approved else "Rejected"


@lru_cache(maxsize=4)
def load_model(path: Path = config.MODEL_PATH) -> Pipeline:
    """Load (and cache) the trained pipeline."""
    if not Path(path).exists():
        raise ModelNotFoundError(
            f"Model artifact not found at {path}. Train it with: python -m loan_approval.train"
        )
    logger.info("Loading model from %s", path)
    return joblib.load(path)


@lru_cache(maxsize=4)
def load_metrics(path: Path = config.METRICS_PATH) -> dict[str, Any]:
    """Load (and cache) the metrics written at training time."""
    if not Path(path).exists():
        raise ModelNotFoundError(f"Metrics not found at {path}. Run: python -m loan_approval.train")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def explain(
    model: Pipeline, X: pd.DataFrame, metrics: dict[str, Any] | None
) -> tuple[str, list[tuple[str, float]]]:
    """Explain a prediction.

    For logistic regression this is exact: each model feature contributes
    ``coef * scaled_value`` to the log-odds (relative to an average applicant,
    since features are standardised). For other models we fall back to the
    global permutation importances recorded at training time, and say so.
    """
    estimator = model.named_steps["model"]
    if hasattr(estimator, "coef_"):
        pre = model[:-1]
        values = np.asarray(pre.transform(X))[0]
        names = pre.named_steps["preprocess"].get_feature_names_out()
        contrib = estimator.coef_[0] * values
        pairs = [(str(n), float(c)) for n, c in zip(names, contrib, strict=True)]
        return "local", sorted(pairs, key=lambda p: -abs(p[1]))
    if metrics:
        imp = metrics.get("permutation_importance_test_roc_auc", {})
        return "global", [(k, float(v["mean"])) for k, v in imp.items()]
    return "global", []


def predict(
    application: LoanApplication,
    model: Pipeline | None = None,
    metrics: dict[str, Any] | None = None,
) -> PredictionResult:
    """Score one application and return probability, decision and top factors."""
    model = model if model is not None else load_model()
    if metrics is None:
        try:
            metrics = load_metrics()
        except ModelNotFoundError:
            metrics = None
    threshold = float(metrics["decision_threshold"]) if metrics else config.DECISION_THRESHOLD
    X = application.to_frame()
    proba = float(model.predict_proba(X)[0, 1])
    kind, factors = explain(model, X, metrics)
    return PredictionResult(
        approval_probability=proba,
        approved=proba >= threshold,
        threshold=threshold,
        factors=factors,
        explanation_kind=kind,
    )
