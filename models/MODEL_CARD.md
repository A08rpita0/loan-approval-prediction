# Model Card: Loan Approval Classifier

_Generated automatically by `python -m loan_approval.train` on 2026-10-04T12:48:15+00:00._

## Model
- **Type:** `hist_gradient_boosting` with feature set `summed_assets_ratios` (scikit-learn 1.9.1)
- **Artifact:** one sklearn `Pipeline` (feature engineering, one-hot encoding, scaling, model)
- **Decision threshold:** 0.5 on predicted approval probability
- **Selection:** simplest candidate within tie_tolerance of the best mean CV score (cv_roc_auc, tolerance 0.001)

## Data
- `loan_approval_dataset.csv`, 4269 rows (sha256 `4b5cd093d178...`), approval rate 0.622
- Stratified split: 3415 train / 854 test, seed 42

## Cross-validation (5-fold stratified, train split only)
| Candidate | ROC-AUC | F1 | Accuracy |
|---|---|---|---|
| logistic_regression/summed_assets | 0.9666 ± 0.0082 | 0.9312 ± 0.0129 | 0.9142 |
| logistic_regression/separate_assets | 0.9663 ± 0.0082 | 0.9313 ± 0.0129 | 0.9142 |
| logistic_regression/summed_assets_ratios | 0.9678 ± 0.0093 | 0.9257 ± 0.0164 | 0.9075 |
| hist_gradient_boosting/summed_assets | 0.9967 ± 0.0024 | 0.9869 ± 0.0016 | 0.9836 |
| hist_gradient_boosting/separate_assets | 0.9979 ± 0.0014 | 0.9885 ± 0.0032 | 0.9857 |
| hist_gradient_boosting/summed_assets_ratios | 0.9999 ± 0.0001 | 0.9962 ± 0.0024 | 0.9953 |

## Held-out test metrics (n=854)
| Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| 0.9965 | 0.9962 | 0.9981 | 0.9972 | 1.0000 | 1.0000 |

Confusion matrix (rows = actual, columns = predicted; positive class = Approved):

| | Pred. Rejected | Pred. Approved |
|---|---|---|
| **Rejected** | 321 | 2 |
| **Approved** | 1 | 530 |

## Permutation importance (drop in test ROC-AUC, raw input columns)
| Input | Importance |
|---|---|
| cibil_score | 0.4636 ± 0.0113 |
| loan_term | 0.0227 ± 0.0018 |
| loan_amount | 0.0225 ± 0.0024 |
| income_annum | 0.0098 ± 0.0012 |
| luxury_assets_value | 0.0034 ± 0.0007 |
| residential_assets_value | 0.0024 ± 0.0006 |
| commercial_assets_value | 0.0022 ± 0.0011 |
| bank_asset_value | 0.0020 ± 0.0008 |
| no_of_dependents | 0.0000 ± 0.0000 |
| self_employed | -0.0000 ± 0.0000 |
| education | -0.0000 ± 0.0000 |

## Intended use and limitations
Educational demo trained on a public Kaggle dataset. Not validated for real
credit decisions. The data is dominated by one feature (CIBIL score), which is
why scores are high; they do not transfer to real lending portfolios.
No fairness audit has been performed.
