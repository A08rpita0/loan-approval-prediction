"""Loading, cleaning, validation and feature engineering for the loan dataset."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from loan_approval import config

logger = logging.getLogger(__name__)

FEATURE_SETS = ("summed_assets", "separate_assets", "summed_assets_ratios")


class SchemaError(ValueError):
    """Raised when the raw data does not match the expected schema."""


def clean(raw: pd.DataFrame, require_target: bool = True) -> pd.DataFrame:
    """Strip whitespace from column names and string values, then validate.

    The source CSV has leading spaces in headers and categorical values
    (e.g. ``" Graduate"``); everything downstream expects stripped values.

    Args:
        raw: Raw dataframe as read from CSV (or built from user input).
        require_target: Whether ``loan_status`` must be present.

    Returns:
        Cleaned copy with the id column dropped.

    Raises:
        SchemaError: On missing columns, nulls, unknown categories or bad numbers.
    """
    df = raw.copy()
    df.columns = [str(c).strip() for c in df.columns]
    for col in df.columns:
        if df[col].dtype == object or pd.api.types.is_string_dtype(df[col]):
            df[col] = df[col].astype("string").str.strip().astype(object)

    expected = list(config.INPUT_COLUMNS) + ([config.TARGET_COLUMN] if require_target else [])
    missing = [c for c in expected if c not in df.columns]
    if missing:
        raise SchemaError(f"Missing required columns: {missing}")
    df = df.drop(columns=[config.ID_COLUMN], errors="ignore")

    if df[expected].isna().any().any():
        bad = df[expected].columns[df[expected].isna().any()].tolist()
        raise SchemaError(f"Null values found in columns: {bad}")

    for col, allowed in config.ALLOWED_CATEGORIES.items():
        unknown = sorted(set(df[col]) - set(allowed))
        if unknown:
            raise SchemaError(f"Unknown values in '{col}': {unknown}; allowed: {allowed}")
    if require_target:
        unknown = sorted(set(df[config.TARGET_COLUMN]) - set(config.TARGET_MAPPING))
        if unknown:
            raise SchemaError(f"Unknown target values: {unknown}")

    for col in config.NUMERIC_INPUT_COLUMNS:
        converted = pd.to_numeric(df[col], errors="coerce")
        if converted.isna().any():
            raise SchemaError(f"Non-numeric values in '{col}'")
        df[col] = converted
    return df[expected].reset_index(drop=True)


def load_dataset(path: Path = config.DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load and clean the CSV, returning raw input columns ``X`` and binary target ``y``."""
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    df = clean(pd.read_csv(path))
    logger.info("Loaded %d rows from %s", len(df), path)
    y = df[config.TARGET_COLUMN].map(config.TARGET_MAPPING).astype(int).rename("approved")
    return df[config.INPUT_COLUMNS], y


def build_features(X: pd.DataFrame, feature_set: str = "summed_assets") -> pd.DataFrame:
    """Derive model features from raw input columns.

    Runs inside the saved sklearn Pipeline (via ``FunctionTransformer``), so the
    exact same transformation is applied at training and prediction time.

    Feature sets:
        * ``summed_assets``: the four asset columns collapsed into ``total_assets``
          (the approach used in the exploratory notebook).
        * ``separate_assets``: the four asset columns kept as-is.
        * ``summed_assets_ratios``: ``summed_assets`` plus loan-to-income and
          loan-to-assets ratios.
    """
    if feature_set not in FEATURE_SETS:
        raise ValueError(f"Unknown feature_set {feature_set!r}; choose from {FEATURE_SETS}")
    out = X.copy()
    if feature_set == "separate_assets":
        return out
    out["total_assets"] = out[config.ASSET_COLUMNS].sum(axis=1)
    out = out.drop(columns=config.ASSET_COLUMNS)
    if feature_set == "summed_assets_ratios":
        out["loan_to_income"] = out["loan_amount"] / out["income_annum"].clip(lower=1)
        out["loan_to_assets"] = out["loan_amount"] / out["total_assets"].clip(lower=1)
    return out
