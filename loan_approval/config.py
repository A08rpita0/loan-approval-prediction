"""Project-wide constants: paths, schema, features and training settings."""

from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_PATH: Path = PROJECT_ROOT / "data" / "loan_approval_dataset.csv"
MODELS_DIR: Path = PROJECT_ROOT / "models"
# Optional override, e.g. to point the app at an experimental artifact.
MODEL_PATH: Path = Path(os.environ.get("LOAN_MODEL_PATH", MODELS_DIR / "loan_model.joblib"))
METRICS_PATH: Path = MODEL_PATH.with_name("metrics.json")
MODEL_CARD_PATH: Path = MODEL_PATH.with_name("MODEL_CARD.md")

RANDOM_SEED: int = 42
TEST_SIZE: float = 0.2
CV_FOLDS: int = 5
DECISION_THRESHOLD: float = 0.5

ID_COLUMN = "loan_id"
TARGET_COLUMN = "loan_status"
TARGET_MAPPING: dict[str, int] = {"Approved": 1, "Rejected": 0}

ASSET_COLUMNS: list[str] = [
    "residential_assets_value",
    "commercial_assets_value",
    "luxury_assets_value",
    "bank_asset_value",
]
NUMERIC_INPUT_COLUMNS: list[str] = [
    "no_of_dependents",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    *ASSET_COLUMNS,
]
CATEGORICAL_INPUT_COLUMNS: list[str] = ["education", "self_employed"]
# Raw applicant columns the model pipeline accepts (feature engineering happens inside it).
INPUT_COLUMNS: list[str] = [
    "no_of_dependents",
    "education",
    "self_employed",
    "income_annum",
    "loan_amount",
    "loan_term",
    "cibil_score",
    *ASSET_COLUMNS,
]
ALLOWED_CATEGORIES: dict[str, list[str]] = {
    "education": ["Graduate", "Not Graduate"],
    "self_employed": ["No", "Yes"],
}

# Validation ranges for single applications (inclusive).
CIBIL_RANGE: tuple[int, int] = (300, 900)
DEPENDENTS_RANGE: tuple[int, int] = (0, 20)
LOAN_TERM_RANGE: tuple[int, int] = (1, 30)  # years
