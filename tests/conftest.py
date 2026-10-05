"""Shared fixtures."""

from __future__ import annotations

import pytest
from threadpoolctl import threadpool_limits

from loan_approval import config
from loan_approval.predict import LoanApplication


@pytest.fixture(autouse=True, scope="session")
def _single_thread():
    # Avoid OpenMP oversubscription slowdowns with HistGradientBoosting.
    with threadpool_limits(limits=1):
        yield


@pytest.fixture
def base_application() -> LoanApplication:
    return LoanApplication(
        no_of_dependents=2,
        education="Graduate",
        self_employed="No",
        income_annum=5_000_000,
        loan_amount=15_000_000,
        loan_term=10,
        cibil_score=650,
        residential_assets_value=7_000_000,
        commercial_assets_value=5_000_000,
        luxury_assets_value=15_000_000,
        bank_asset_value=5_000_000,
    )


requires_model = pytest.mark.skipif(
    not config.MODEL_PATH.exists(),
    reason="trained artifact missing; run python -m loan_approval.train",
)
