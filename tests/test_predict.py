"""Tests for input validation, encoding and inference behaviour."""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from loan_approval.data import load_dataset
from loan_approval.predict import LoanApplication, load_model, predict
from loan_approval.train import build_pipeline
from tests.conftest import requires_model


@pytest.mark.parametrize(
    "change",
    [
        {"income_annum": -1},
        {"cibil_score": 299},
        {"cibil_score": 901},
        {"loan_amount": 0},
        {"education": "Graduated"},  # not a dataset label (dataset uses "Graduate")
        {"self_employed": "maybe"},
        {"luxury_assets_value": -5},
    ],
)
def test_application_validation_rejects_bad_input(base_application, change):
    with pytest.raises(ValueError):
        dataclasses.replace(base_application, **change)


class _RecordingModel:
    """Stand-in model that records the frame it receives."""

    def __init__(self):
        self.seen = None
        self.named_steps = {"model": object()}

    def predict_proba(self, X):
        self.seen = X.copy()
        return np.array([[0.3, 0.7]])


def test_categories_reach_the_model_as_raw_strings(base_application):
    """Regression test: categories must not be hand-encoded before the pipeline.

    Inference must hand the pipeline the same string labels used in training.
    """
    model = _RecordingModel()
    for label in ("Graduate", "Not Graduate"):
        predict(dataclasses.replace(base_application, education=label), model=model, metrics=None)
        assert model.seen.loc[0, "education"] == label


def test_pipeline_encodes_education_like_training():
    X, y = load_dataset()
    pipe = build_pipeline("logistic_regression", "summed_assets").fit(X, y)
    encoded = pipe[:-1].transform(X.head(50))
    expected = (X.head(50)["education"] == "Not Graduate").astype(float).to_numpy()
    np.testing.assert_array_equal(encoded["education_Not Graduate"].to_numpy(), expected)


def test_logistic_explanation_is_local_and_exact():
    X, y = load_dataset()
    pipe = build_pipeline("logistic_regression", "summed_assets").fit(X, y)
    row = X.iloc[0]
    app = LoanApplication(**{k: (v.item() if hasattr(v, "item") else v) for k, v in row.items()})
    result = predict(app, model=pipe, metrics={"decision_threshold": 0.5})
    assert result.explanation_kind == "local"
    assert "cibil_score" in [n for n, _ in result.factors]
    # Contributions plus intercept reproduce the predicted log-odds.
    logit = sum(v for _, v in result.factors) + pipe.named_steps["model"].intercept_[0]
    assert 1 / (1 + np.exp(-logit)) == pytest.approx(result.approval_probability)


@requires_model
def test_trained_model_probability_in_range(base_application):
    result = predict(base_application)
    assert 0.0 <= result.approval_probability <= 1.0
    assert result.label in {"Approved", "Rejected"}


@requires_model
def test_higher_cibil_never_lowers_approval_probability():
    """Monotonic sanity check on the real artifact (enforced via monotonic_cst)."""
    X, _ = load_dataset()
    model = load_model()
    scores = np.arange(300, 901, 10)
    for i in range(0, len(X), 50):
        grid = X.iloc[[i] * len(scores)].copy()
        grid["cibil_score"] = scores
        proba = model.predict_proba(grid)[:, 1]
        assert np.all(np.diff(proba) >= -1e-12), f"non-monotonic for row {i}"
