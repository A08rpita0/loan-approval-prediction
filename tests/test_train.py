"""End-to-end training on a small subsample."""

from __future__ import annotations

import json

import joblib

from loan_approval import config
from loan_approval.train import run


def test_training_writes_artifact_with_expected_keys(tmp_path):
    metrics = run(out_dir=tmp_path, sample=600)
    assert (tmp_path / "loan_model.joblib").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()
    on_disk = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert on_disk == json.loads(json.dumps(metrics))
    for key in (
        "chosen_model",
        "feature_set",
        "decision_threshold",
        "input_columns",
        "cv",
        "test",
        "data",
        "sklearn_version",
        "trained_at",
        "permutation_importance_test_roc_auc",
    ):
        assert key in metrics
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc", "confusion_matrix"):
        assert key in metrics["test"]
    assert metrics["data"]["n_rows"] == 600
    assert metrics["data"]["sampled"]
    assert metrics["input_columns"] == config.INPUT_COLUMNS
    assert hasattr(joblib.load(tmp_path / "loan_model.joblib"), "predict_proba")
