"""
Feature definitions for the Telco Customer Churn dataset.

This module defines which columns belong to which category.
Keeping this separate makes it easy to modify feature sets
without changing pipeline code.

Column Categories:
- IDENTIFIER: Should be dropped before modeling
- TARGET: The variable we're predicting
- NUMERIC: Continuous or discrete numeric features
- BINARY: Yes/No features (will be encoded as 0/1)
- CATEGORICAL: Multi-class categorical features (will be one-hot encoded)
"""

# Identifier column - must be dropped before modeling
IDENTIFIER_COLS = ['customerID']

# Target variable
TARGET_COL = 'Churn'

# Numeric features
# These will be scaled using StandardScaler
NUMERIC_COLS = [
    'tenure',          # Months with company (0-72)
    'MonthlyCharges',  # Current monthly charge
    'TotalCharges',    # Total amount charged to date
]

# Binary features (Yes/No or 0/1)
# These will be encoded as 0/1
BINARY_COLS = [
    'gender',           # Male/Female
    'SeniorCitizen',    # Already 0/1 in raw data
    'Partner',          # Yes/No
    'Dependents',       # Yes/No
    'PhoneService',     # Yes/No
    'PaperlessBilling', # Yes/No
]

# Categorical features (multiple categories)
# These will be one-hot encoded
CATEGORICAL_COLS = [
    'MultipleLines',     # Yes/No/No phone service
    'InternetService',   # DSL/Fiber optic/No
    'OnlineSecurity',    # Yes/No/No internet service
    'OnlineBackup',      # Yes/No/No internet service
    'DeviceProtection',  # Yes/No/No internet service
    'TechSupport',       # Yes/No/No internet service
    'StreamingTV',       # Yes/No/No internet service
    'StreamingMovies',   # Yes/No/No internet service
    'Contract',          # Month-to-month/One year/Two year
    'PaymentMethod',     # Electronic check/Mailed check/Bank transfer/Credit card
]

# All feature columns (excludes identifier and target)
FEATURE_COLS = NUMERIC_COLS + BINARY_COLS + CATEGORICAL_COLS


def get_feature_columns() -> list:
    """Return all feature column names."""
    return FEATURE_COLS.copy()


def get_column_groups() -> dict:
    """Return a dictionary of column groups for pipeline construction."""
    return {
        'numeric': NUMERIC_COLS.copy(),
        'binary': BINARY_COLS.copy(),
        'categorical': CATEGORICAL_COLS.copy()
    }
