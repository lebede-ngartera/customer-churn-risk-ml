"""
Preprocessing module for customer churn prediction.

This module handles all data loading, cleaning, and preprocessing steps.
It builds a scikit-learn ColumnTransformer that can be embedded in a Pipeline
to prevent data leakage between train/val/test splits.

Design decisions:
- TotalCharges has blank strings for new customers (tenure=0); we convert to NaN
  and let the imputer handle it (median imputation).
- Categorical variables use one-hot encoding with handle_unknown='ignore' so
  unseen categories at inference time don't crash the model.
- Numeric features are scaled with StandardScaler after imputation.
"""

import logging
from pathlib import Path
from typing import Tuple, List, Optional

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Feature Definitions
# =============================================================================
# These lists define which columns get which preprocessing treatment.
# Keeping them explicit makes the pipeline transparent and auditable.

NUMERIC_FEATURES: List[str] = [
    'tenure',
    'MonthlyCharges', 
    'TotalCharges'
]

CATEGORICAL_FEATURES: List[str] = [
    'gender',
    'SeniorCitizen',      # 0/1 but treat as categorical for clarity
    'Partner',
    'Dependents',
    'PhoneService',
    'MultipleLines',
    'InternetService',
    'OnlineSecurity',
    'OnlineBackup',
    'DeviceProtection',
    'TechSupport',
    'StreamingTV',
    'StreamingMovies',
    'Contract',
    'PaperlessBilling',
    'PaymentMethod'
]

# Columns to drop (identifiers, not features)
DROP_COLUMNS: List[str] = ['customerID']

# Target column
TARGET_COLUMN: str = 'Churn'


# =============================================================================
# Data Loading and Cleaning
# =============================================================================

def load_raw_data(filepath: str = 'data/raw/telco_customer_churn.csv') -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset from CSV.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        Raw DataFrame as-is from the CSV.
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
        pd.errors.EmptyDataError: If the CSV is empty.
    """
    path = Path(filepath)
    
    if not path.exists():
        raise FileNotFoundError(
            f"Data file not found at '{filepath}'. "
            f"Please download the IBM Telco Customer Churn dataset and place it there."
        )
    
    logger.info(f"Loading data from {filepath}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the raw data: strip whitespace, fix types, drop identifiers.
    
    Key cleaning steps:
    1. Strip whitespace from string columns
    2. Convert TotalCharges from string to numeric (blanks become NaN)
    3. Drop customerID (identifier, not a feature)
    
    Args:
        df: Raw DataFrame from load_raw_data().
        
    Returns:
        Cleaned DataFrame ready for preprocessing.
    """
    df = df.copy()
    
    # Strip whitespace from all string columns
    # This catches issues like " Yes" vs "Yes"
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = df[col].str.strip()
    
    # TotalCharges is stored as string with blanks for new customers
    # Convert to numeric; blanks become NaN (handled by imputer later)
    # Why not fill with 0? Because 0 has meaning (free service), NaN is "unknown"
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    n_missing = df['TotalCharges'].isna().sum()
    if n_missing > 0:
        logger.info(f"TotalCharges: {n_missing} blank values converted to NaN (will be imputed)")
    
    # Drop identifier column - it has no predictive value
    if 'customerID' in df.columns:
        df = df.drop(columns=['customerID'])
        logger.info("Dropped customerID column (identifier)")
    
    return df


def encode_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Separate features and encode the target variable.
    
    Target encoding:
    - "Yes" -> 1 (churned)
    - "No"  -> 0 (retained)
    
    Args:
        df: Cleaned DataFrame with Churn column.
        
    Returns:
        Tuple of (X features DataFrame, y target Series).
        
    Raises:
        ValueError: If Churn column is missing or has unexpected values.
    """
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Target column '{TARGET_COLUMN}' not found in DataFrame")
    
    # Encode target: Yes=1 (churn), No=0 (stay)
    target_map = {'Yes': 1, 'No': 0}
    y = df[TARGET_COLUMN].map(target_map)
    
    # Check for unmapped values
    if y.isna().any():
        unexpected = df[TARGET_COLUMN][y.isna()].unique()
        raise ValueError(f"Unexpected values in Churn column: {unexpected}")
    
    y = y.astype(int)
    
    # Separate features
    X = df.drop(columns=[TARGET_COLUMN])
    
    logger.info(f"Target distribution: {y.value_counts().to_dict()}")
    logger.info(f"Churn rate: {y.mean():.1%}")
    
    return X, y


# =============================================================================
# Preprocessor Construction
# =============================================================================

def build_numeric_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for numeric features.
    
    Steps:
    1. Impute missing values with median (robust to outliers)
    2. Scale to zero mean, unit variance (important for logistic regression)
    
    Why median imputation?
    - TotalCharges has missing values for tenure=0 customers
    - Median is robust to the right-skewed distribution of charges
    
    Why StandardScaler?
    - Logistic regression converges faster with scaled features
    - Coefficients become comparable across features
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def build_categorical_pipeline() -> Pipeline:
    """
    Build preprocessing pipeline for categorical features.
    
    Steps:
    1. Impute missing values with most frequent (mode)
    2. One-hot encode with handle_unknown='ignore'
    
    Why handle_unknown='ignore'?
    - At inference time, we might see categories not in training data
    - 'ignore' creates all-zero encoding instead of raising an error
    - This is safer for production deployment
    
    Why drop='first' is NOT used?
    - Logistic regression with regularization handles collinearity
    - Keeping all dummies makes feature importance more interpretable
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])


def build_preprocessor() -> ColumnTransformer:
    """
    Build the full preprocessing ColumnTransformer.
    
    This transformer applies different preprocessing to different column types:
    - Numeric: impute + scale
    - Categorical: impute + one-hot encode
    
    The ColumnTransformer ensures no data leakage when used in a Pipeline:
    - fit() learns parameters (means, categories) from training data only
    - transform() applies those parameters to any data
    
    Returns:
        Unfitted ColumnTransformer ready to be used in a Pipeline.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', build_numeric_pipeline(), NUMERIC_FEATURES),
            ('cat', build_categorical_pipeline(), CATEGORICAL_FEATURES)
        ],
        # If a column isn't in either list, drop it (safety net)
        remainder='drop',
        # Preserve feature names for explainability
        verbose_feature_names_out=True
    )
    
    logger.info(f"Built preprocessor with {len(NUMERIC_FEATURES)} numeric "
                f"and {len(CATEGORICAL_FEATURES)} categorical features")
    
    return preprocessor


# =============================================================================
# Main Entry Point
# =============================================================================

def prepare_data(
    filepath: str = 'data/raw/telco_customer_churn.csv'
) -> Tuple[pd.DataFrame, pd.Series, ColumnTransformer]:
    """
    Load, clean, and prepare data for modeling.
    
    This is the main entry point that combines all preprocessing steps.
    
    Args:
        filepath: Path to the raw CSV file.
        
    Returns:
        Tuple of:
        - X: Feature DataFrame (cleaned but not transformed)
        - y: Target Series (binary encoded)
        - preprocessor: Unfitted ColumnTransformer
        
    Example:
        >>> X, y, preprocessor = prepare_data()
        >>> X_train, X_test, y_train, y_test = train_test_split(X, y)
        >>> pipeline = Pipeline([('prep', preprocessor), ('clf', LogisticRegression())])
        >>> pipeline.fit(X_train, y_train)
    """
    # Load raw data
    df = load_raw_data(filepath)
    
    # Clean data (fix types, strip whitespace, drop ID)
    df = clean_data(df)
    
    # Separate features and target
    X, y = encode_target(df)
    
    # Build preprocessor (unfitted - will be fitted in pipeline)
    preprocessor = build_preprocessor()
    
    logger.info(f"Data prepared: X shape {X.shape}, y shape {y.shape}")
    
    return X, y, preprocessor


def get_feature_names(preprocessor: ColumnTransformer) -> List[str]:
    """
    Get feature names after transformation.
    
    Note: The preprocessor must be fitted first!
    
    Args:
        preprocessor: A fitted ColumnTransformer.
        
    Returns:
        List of feature names in the transformed output.
    """
    return list(preprocessor.get_feature_names_out())


def get_input_schema() -> dict:
    """
    Get the expected input schema for the model.
    
    This is useful for API documentation and validation.
    
    Returns:
        Dictionary describing expected features and their types.
    """
    schema = {
        'numeric_features': NUMERIC_FEATURES,
        'categorical_features': CATEGORICAL_FEATURES,
        'all_features': NUMERIC_FEATURES + CATEGORICAL_FEATURES,
        'target': TARGET_COLUMN
    }
    return schema


if __name__ == '__main__':
    # Quick test of the preprocessing pipeline
    print("Testing preprocessing module...")
    
    try:
        X, y, preprocessor = prepare_data()
        print(f"\n✓ Data loaded successfully")
        print(f"  Features: {X.shape[1]}")
        print(f"  Samples: {X.shape[0]}")
        print(f"  Churn rate: {y.mean():.1%}")
        
        # Test that preprocessor can be fitted
        preprocessor.fit(X)
        feature_names = get_feature_names(preprocessor)
        print(f"  Transformed features: {len(feature_names)}")
        
        print("\n✓ Preprocessing module working correctly")
        
    except FileNotFoundError as e:
        print(f"\n✗ {e}")
        print("  Download the dataset first.")
