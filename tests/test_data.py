"""Tests for loading, cleaning and feature engineering."""

from __future__ import annotations

import pandas as pd
import pytest

from loan_approval import config
from loan_approval.data import SchemaError, build_features, clean, load_dataset


def _raw_row(**overrides) -> dict:
    row = {
        " loan_id": 1,
        " no_of_dependents": 2,
        " education": " Graduate",
        " self_employed": " No",
        " income_annum": 9600000,
        " loan_amount": 29900000,
        " loan_term": 12,
        " cibil_score": 778,
        " residential_assets_value": 2400000,
        " commercial_assets_value": 17600000,
        " luxury_assets_value": 22700000,
        " bank_asset_value": 8000000,
        " loan_status": " Approved",
    }
    row.update(overrides)
    return row


def test_clean_strips_whitespace_from_headers_and_values():
    df = clean(pd.DataFrame([_raw_row()]))
    assert "loan_id" not in df.columns
    assert list(df.columns) == [*config.INPUT_COLUMNS, config.TARGET_COLUMN]
    assert df.loc[0, "education"] == "Graduate"
    assert df.loc[0, "self_employed"] == "No"
    assert df.loc[0, "loan_status"] == "Approved"


def test_clean_rejects_missing_column():
    raw = pd.DataFrame([_raw_row()]).drop(columns=[" cibil_score"])
    with pytest.raises(SchemaError, match="cibil_score"):
        clean(raw)


def test_clean_rejects_unknown_category():
    with pytest.raises(SchemaError, match="education"):
        clean(pd.DataFrame([_raw_row(**{" education": " PhD"})]))


def test_clean_rejects_non_numeric():
    with pytest.raises(SchemaError, match="income_annum"):
        clean(pd.DataFrame([_raw_row(**{" income_annum": "lots"})]))


def test_load_dataset_maps_target():
    X, y = load_dataset()
    assert len(X) == 4269
    assert set(y.unique()) == {0, 1}
    assert set(X["education"].unique()) == {"Graduate", "Not Graduate"}


@pytest.mark.parametrize(
    ("feature_set", "expected_new", "dropped_assets"),
    [
        ("summed_assets", {"total_assets"}, True),
        ("separate_assets", set(), False),
        ("summed_assets_ratios", {"total_assets", "loan_to_income", "loan_to_assets"}, True),
    ],
)
def test_build_features(feature_set, expected_new, dropped_assets):
    X = clean(pd.DataFrame([_raw_row()]))[config.INPUT_COLUMNS]
    out = build_features(X, feature_set)
    assert expected_new <= set(out.columns)
    assert all((c not in out.columns) == dropped_assets for c in config.ASSET_COLUMNS)
    if "total_assets" in out:
        assert out.loc[0, "total_assets"] == 2400000 + 17600000 + 22700000 + 8000000
