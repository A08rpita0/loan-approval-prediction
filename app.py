"""Streamlit front end for the loan approval model.

Run with: streamlit run app.py
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from loan_approval import config
from loan_approval.predict import (
    LoanApplication,
    ModelNotFoundError,
    load_metrics,
    load_model,
    predict,
)

st.set_page_config(page_title="Loan Approval Prediction", page_icon=":bank:", layout="centered")

FACTOR_LABELS = {
    "no_of_dependents": "Number of dependents",
    "education": "Education",
    "education_Not Graduate": "Education: not graduate",
    "self_employed": "Self-employed",
    "self_employed_Yes": "Self-employed: yes",
    "income_annum": "Annual income",
    "loan_amount": "Loan amount",
    "loan_term": "Loan term",
    "cibil_score": "CIBIL score",
    "total_assets": "Total assets",
    "residential_assets_value": "Residential assets",
    "commercial_assets_value": "Commercial assets",
    "luxury_assets_value": "Luxury assets",
    "bank_asset_value": "Bank assets",
    "loan_to_income": "Loan-to-income ratio",
    "loan_to_assets": "Loan-to-assets ratio",
}


@st.cache_resource
def get_model():
    return load_model()


@st.cache_data
def get_metrics() -> dict:
    return load_metrics()


st.title("Loan Approval Prediction")
st.caption(
    "Estimates the probability that a loan application is approved, using a model "
    "trained on a public Kaggle dataset. Educational demo, not a credit decision."
)

try:
    model = get_model()
    metrics = get_metrics()
except ModelNotFoundError as exc:
    st.error(f"{exc}")
    st.code("python -m loan_approval.train", language="bash")
    st.stop()

with st.form("application"):
    st.subheader("Applicant")
    c1, c2 = st.columns(2)
    dependents = c1.number_input("Number of dependents", min_value=0, max_value=20, value=2, step=1)
    education = c2.selectbox("Education", config.ALLOWED_CATEGORIES["education"])
    self_employed = c1.selectbox("Self-employed", config.ALLOWED_CATEGORIES["self_employed"])
    income = c2.number_input("Annual income (INR)", min_value=0, value=5_000_000, step=100_000)

    st.subheader("Loan")
    c1, c2, c3 = st.columns(3)
    loan_amount = c1.number_input("Loan amount (INR)", min_value=0, value=15_000_000, step=100_000)
    loan_term = c2.number_input("Loan term (years)", min_value=1, max_value=30, value=10, step=1)
    cibil = c3.number_input("CIBIL score", min_value=300, max_value=900, value=650, step=1)

    st.subheader("Assets (INR)")
    c1, c2 = st.columns(2)
    residential = c1.number_input("Residential", min_value=0, value=7_000_000, step=100_000)
    commercial = c2.number_input("Commercial", min_value=0, value=5_000_000, step=100_000)
    luxury = c1.number_input("Luxury", min_value=0, value=15_000_000, step=100_000)
    bank = c2.number_input("Bank", min_value=0, value=5_000_000, step=100_000)

    submitted = st.form_submit_button("Predict", type="primary", use_container_width=True)

if submitted:
    try:
        application = LoanApplication(
            no_of_dependents=int(dependents),
            education=education,
            self_employed=self_employed,
            income_annum=float(income),
            loan_amount=float(loan_amount),
            loan_term=int(loan_term),
            cibil_score=int(cibil),
            residential_assets_value=float(residential),
            commercial_assets_value=float(commercial),
            luxury_assets_value=float(luxury),
            bank_asset_value=float(bank),
        )
    except ValueError as exc:
        st.error(f"Invalid input: {exc}")
        st.stop()

    result = predict(application, model=model, metrics=metrics)
    st.divider()
    m1, m2 = st.columns(2)
    m1.metric("Approval probability", f"{result.approval_probability:.1%}")
    m2.metric("Decision", result.label, help=f"Approved if probability >= {result.threshold:.2f}")
    if result.approved:
        st.success("Likely to be approved.")
    else:
        st.error("Likely to be rejected.")

    top = result.factors[:5]
    if top:
        df = pd.DataFrame(
            {"Factor": [FACTOR_LABELS.get(n, n) for n, _ in top], "Value": [v for _, v in top]}
        )
        if result.explanation_kind == "local":
            st.subheader("Top factors for this application")
            st.caption(
                "Contribution to the log-odds of approval relative to an average applicant "
                "(positive pushes towards approval)."
            )
        else:
            st.subheader("Most important inputs overall")
            st.caption(
                "Global permutation importance from training (drop in test ROC-AUC when the "
                "input is shuffled). Not specific to this application."
            )
        st.bar_chart(df, x="Factor", y="Value", horizontal=True)

with st.expander("Model details and evaluation"):
    t = metrics["test"]
    st.write(
        f"Model: `{metrics['chosen_model']}` (feature set `{metrics['feature_set']}`), "
        f"trained {metrics['trained_at']} with scikit-learn {metrics['sklearn_version']}. "
        f"Held-out test set: {metrics['data']['n_test']} applications."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Metric": ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC", "PR-AUC"],
                "Test": [
                    t[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc", "pr_auc")
                ],
            }
        ).style.format({"Test": "{:.4f}"}),
        hide_index=True,
    )
    cm = t["confusion_matrix"]
    st.write("Confusion matrix (rows: actual, columns: predicted)")
    st.dataframe(
        pd.DataFrame(
            [[cm["tn"], cm["fp"]], [cm["fn"], cm["tp"]]],
            index=["Actual rejected", "Actual approved"],
            columns=["Predicted rejected", "Predicted approved"],
        )
    )
