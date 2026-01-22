# Customer Churn Prediction

A production-minded machine learning system for predicting customer churn risk in telecommunications.

## Problem Statement

Predict the probability that a customer will churn in the next billing cycle using historical customer account and service data.

- **Task**: Binary classification (churn = 0 / 1)
- **Dataset**: IBM Telco Customer Churn (public, tabular)
- **Approach**: Classical ML with scikit-learn, pipeline-based preprocessing

## Project Principles

1. **No data leakage** — strict train/validation/test separation
2. **Reproducibility** — fixed random seeds, versioned dependencies
3. **Interpretability** — start simple, explain all decisions
4. **Maintainability** — clean code that another engineer can read and modify

## Model Performance

### Baseline Model (Logistic Regression)

| Metric | Validation | Test |
|--------|------------|------|
| ROC-AUC | ~0.84 | ~0.84 |
| Precision | ~0.65 | ~0.65 |
| Recall | ~0.54 | ~0.54 |
| F1 | ~0.59 | ~0.59 |

### Metric Choice Rationale

- **ROC-AUC (primary)**: Measures ranking ability across all thresholds. Ideal for imbalanced data because it's threshold-independent.
- **Precision**: Of customers predicted to churn, how many actually do? High precision = fewer wasted retention offers.
- **Recall**: Of customers who actually churn, how many do we catch? High recall = fewer missed churners.
- **F1**: Harmonic mean of precision and recall. Useful summary, but hides the tradeoff.

**Business Context**: The cost of a false negative (missed churner, ~$500 CLV lost) is much higher than a false positive (unnecessary retention offer, ~$50). This asymmetry suggests optimizing for recall over precision in production.

## Modeling Approach

The baseline model uses **Logistic Regression** for the following reasons:

1. **Interpretability**: Coefficients directly indicate feature importance and direction. A positive coefficient means the feature increases churn probability.
2. **Speed**: Fast to train and predict, suitable for batch and real-time inference.
3. **Regularization**: L2 penalty (default) prevents overfitting on high-dimensional one-hot encoded features.
4. **Baseline benchmark**: Provides a clear performance floor before exploring more complex models.

The model is wrapped in a scikit-learn `Pipeline` with a `ColumnTransformer`:

```
Pipeline
├── ColumnTransformer (preprocessing)
│   ├── Numeric: SimpleImputer(median) → StandardScaler
│   └── Categorical: SimpleImputer(most_frequent) → OneHotEncoder
└── LogisticRegression(class_weight='balanced')
```

This design ensures:
- **No data leakage**: The preprocessor is fit only on training data.
- **Reproducibility**: The entire pipeline is serialized as a single artifact.
- **Portability**: One object handles both preprocessing and prediction.

## Evaluation Strategy

### Primary Metric: ROC-AUC

ROC-AUC measures the model's ability to rank churners higher than non-churners across all classification thresholds. It is threshold-independent, making it suitable for imbalanced datasets where the optimal threshold may not be 0.5.

### Secondary Metrics

| Metric | Purpose | Tradeoff |
|--------|---------|----------|
| **Recall** | Catch as many churners as possible | Higher recall → more false positives |
| **Precision** | Avoid wasting retention offers | Higher precision → more missed churners |
| **F1** | Balance precision and recall | Hides the underlying tradeoff |

### Why Recall Over Precision

In this business context:
- **False negative cost**: ~$500 (lost customer lifetime value)
- **False positive cost**: ~$50 (wasted retention offer)

The 10:1 cost asymmetry means missing a churner is far more expensive than a false alarm. The model is tuned to favor recall, accepting lower precision to reduce missed churners.

### Why Accuracy Is De-emphasized

With ~26% churn rate, a naive model predicting "no churn" for everyone achieves 74% accuracy while catching zero churners. Accuracy is misleading for imbalanced classification and is reported only for completeness.

## Phase 9: Model Review & Analysis

Phase 9 focuses on understanding model behavior beyond aggregate metrics. This includes threshold tuning, probability calibration, and systematic error analysis.

### Generated Reports

| Report | Description |
|--------|-------------|
| `reports/threshold_analysis.md` | Precision-recall tradeoffs at different thresholds |
| `reports/calibration.md` | Reliability diagram and Brier score analysis |
| `reports/error_analysis.md` | Patterns in false positives and false negatives |

These reports help answer:
- What threshold should we use in production?
- Are predicted probabilities trustworthy?
- Which customer segments does the model struggle with?

## Data Leakage & Assumptions

### Prediction Time Definition

The model predicts whether a customer will churn **before their next billing cycle**. All features must be available at prediction time without knowledge of future events.

### Leakage Prevention

1. **Identifier columns dropped**: `customerID` is removed during preprocessing. It has no predictive value and could cause spurious correlations.

2. **Preprocessing fit on training data only**: The `ColumnTransformer` learns imputation values and encoding categories exclusively from the training set. Validation and test sets are transformed using these learned parameters.

3. **Stratified splitting**: Train/validation/test splits preserve the class distribution to avoid biased evaluation.

### Known Leakage Risks

| Feature | Risk | Mitigation |
|---------|------|------------|
| `TotalCharges` | Correlated with `tenure` (longer tenure → higher total charges) | Kept as-is; the correlation is legitimate and available at prediction time |
| `Churn` label timing | If churn is recorded mid-cycle, features may reflect post-decision behavior | Assumed the dataset represents end-of-cycle snapshots |

### Assumptions

- The dataset is a static snapshot; no temporal ordering is preserved.
- Feature values represent the state at the end of the billing cycle.
- The churn label indicates whether the customer left after that cycle.

## Artifacts

Training produces three artifacts in the `artifacts/` directory:

| File | Format | Purpose |
|------|--------|---------|
| `model.joblib` | Pickle (joblib) | Fitted sklearn Pipeline for inference. Contains both preprocessing and classifier. |
| `metrics.json` | JSON | Validation and test metrics, model configuration, and training timestamp. Used for model versioning and monitoring. |
| `schema.json` | JSON | Input feature names, types, and transformed feature names. Used for API input validation and documentation. |

### Loading the Model

```python
import joblib

pipeline = joblib.load('artifacts/model.joblib')
predictions = pipeline.predict(X_new)
probabilities = pipeline.predict_proba(X_new)[:, 1]
```

## How to Reproduce Results

All commands should be run from the project root directory.

```bash
# 1. Install dependencies
pip install -e .

# 2. Train the baseline model
python src/train.py

# 3. Run model analysis (threshold tuning, calibration)
python src/analysis.py

# 4. Run error analysis
python src/error_analysis.py

# 5. Run all tests
pytest tests/ -v
```

Training uses a fixed random seed (`RANDOM_STATE=42`) to ensure identical results across runs.

## Limitations & Future Work

### Current Limitations

1. **Static snapshot data**: The model is trained on a single point-in-time dataset. Customer behavior evolves, and the model may degrade without retraining.

2. **No temporal modeling**: Churn is inherently a time-to-event problem. The current approach ignores when customers churn, only whether they do.

3. **No intervention feedback**: The model cannot learn from retention campaign outcomes. Customers who received offers and stayed are indistinguishable from those who would have stayed anyway.

4. **Calibration**: Predicted probabilities may not perfectly match observed churn rates. The calibration report identifies where adjustments are needed.

### Future Improvements

- **Survival analysis**: Model time-to-churn using Cox regression or Kaplan-Meier estimators.
- **Uplift modeling**: Predict which customers would respond to retention offers.
- **Online learning**: Update the model incrementally as new data arrives.
- **Feature engineering**: Add derived features like tenure buckets, service bundles, or payment consistency.
- **Calibration tuning**: Apply Platt scaling or isotonic regression to improve probability estimates.

## Repository Structure

```
├── data/
│   ├── raw/              # Original, immutable data (not committed)
│   └── processed/        # Cleaned data after validation
├── notebooks/            # Numbered exploration notebooks
│   ├── 01_data_exploration.ipynb
│   ├── 02_model_evaluation.ipynb
│   └── 03_explainability.ipynb
├── src/
│   ├── preprocessing.py  # Data loading, cleaning, ColumnTransformer
│   ├── train.py          # CLI training script (main entry point)
│   ├── evaluate.py       # Metrics computation and reporting
│   ├── data/             # Data loading and validation (legacy)
│   ├── features/         # Feature engineering pipelines (legacy)
│   ├── models/           # Model training, evaluation, explainability
│   └── api/              # FastAPI prediction service
├── artifacts/            # Training outputs (not committed)
│   ├── model.joblib      # Fitted sklearn Pipeline
│   ├── metrics.json      # Validation and test metrics
│   └── schema.json       # Input feature schema
├── models/               # Serialized model artifacts (legacy)
├── tests/                # Unit and integration tests (60+ tests)
└── configs/              # Configuration files
```

## Quick Start

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/<username>/customer_churn_prediction.git
cd customer_churn_prediction
pip install -e .
```

### Training the Baseline Model

```bash
# Ensure data is in place
# Download IBM Telco Customer Churn dataset to data/raw/telco_customer_churn.csv

# Run training (from project root)
python src/train.py

# Or with custom paths
python src/train.py --data-path data/raw/telco_customer_churn.csv --output-dir artifacts
```

**What happens during training:**
1. Loads and cleans the raw CSV
2. Creates stratified 70/15/15 train/val/test split
3. Builds preprocessing pipeline (imputation + scaling + one-hot encoding)
4. Trains Logistic Regression with class balancing
5. Evaluates on validation and test sets
6. Saves model, metrics, and schema to `artifacts/`

### Artifacts Produced

| File | Description |
|------|-------------|
| `artifacts/model.joblib` | Full sklearn Pipeline (preprocessor + classifier) |
| `artifacts/metrics.json` | Validation and test metrics with config |
| `artifacts/schema.json` | Input feature names and schema |

### Using the Trained Model

```python
import joblib
import pandas as pd

# Load the pipeline
pipeline = joblib.load('artifacts/model.joblib')

# Make predictions on new data
new_customer = pd.DataFrame([{
    'tenure': 24,
    'MonthlyCharges': 65.5,
    'TotalCharges': 1572.0,
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'Yes',
    'Dependents': 'No',
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'Fiber optic',
    'OnlineSecurity': 'No',
    'OnlineBackup': 'Yes',
    'DeviceProtection': 'No',
    'TechSupport': 'No',
    'StreamingTV': 'Yes',
    'StreamingMovies': 'Yes',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'Yes',
    'PaymentMethod': 'Electronic check'
}])

# Get prediction and probability
prediction = pipeline.predict(new_customer)[0]  # 0 or 1
probability = pipeline.predict_proba(new_customer)[0, 1]  # P(churn)

print(f"Churn prediction: {'Yes' if prediction else 'No'}")
print(f"Churn probability: {probability:.1%}")
```

## API Endpoints

Once the API is running, visit `http://localhost:8000/docs` for interactive documentation.

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/predict` | POST | Single customer prediction |
| `/predict/batch` | POST | Batch predictions |
| `/model/info` | GET | Model metadata |

### Start the API

```bash
PYTHONPATH=. uvicorn src.api.main:app --reload
```

### Example API Request

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "tenure": 24,
    "MonthlyCharges": 65.5,
    "TotalCharges": 1572.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check"
  }'
```

## Development Status

- [x] Phase 1: Repository scaffolding
- [x] Phase 2: Data exploration
- [x] Phase 3: Feature engineering
- [x] Phase 4: Modeling
- [x] Phase 5: Evaluation
- [x] Phase 6: Explainability
- [x] Phase 7: Deployment (API)
- [x] Phase 8: Reproducible baseline
- [x] Phase 9: Model review & analysis

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run specific test file
pytest tests/test_api.py -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html
```

## License

MIT

## Author

[Your Name]
