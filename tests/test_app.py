"""Smoke test the Streamlit app with AppTest."""

from __future__ import annotations

from streamlit.testing.v1 import AppTest

from loan_approval import config
from tests.conftest import requires_model


@requires_model
def test_app_renders_and_predicts():
    at = AppTest.from_file(str(config.PROJECT_ROOT / "app.py"), default_timeout=120)
    at.run()
    assert not at.exception
    cibil = next(n for n in at.number_input if n.label == "CIBIL score")
    cibil.set_value(850)
    at.button[0].click()
    at.run()
    assert not at.exception
    assert "Approval probability" in [m.label for m in at.metric]
    assert any(m.value in {"Approved", "Rejected"} for m in at.metric)
