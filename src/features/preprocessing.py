"""
Preprocessing pipelines for the Telco Customer Churn dataset.

This module builds sklearn Pipelines and ColumnTransformers for
transforming raw features into model-ready format.

Design Principles:
1. All transformations are encapsulated in sklearn Pipelines
2. Pipelines are fitted ONLY on training data
3. The same fitted pipeline is used to transform validation and test data
4. This prevents data leakage from test set statistics
"""

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
import pandas as pd
import numpy as np

from .feature_definitions import (
    NUMERIC_COLS,
    BINARY_COLS, 
    CATEGORICAL_COLS,
    get_column_groups
)


def encode_binary_feature(X: np.ndarray) -> np.ndarray:
    """
    Encode binary features to 0/1.
    
    Handles: Yes/No, Male/Female, and already-numeric (0/1).
    """
    # Convert to string for consistent handling
    X_str = np.array(X, dtype=str)
    
    # Map common binary encodings
    mapping = {
        'yes': 1, 'no': 0,
        'male': 1, 'female': 0,
        '1': 1, '0': 0,
        '1.0': 1, '0.0': 0
    }
    
    result = np.zeros(X_str.shape, dtype=np.float64)
    for i, val in enumerate(X_str.flatten()):
        result.flat[i] = mapping.get(val.lower().strip(), 0)
    
    return result.reshape(X.shape)


def create_numeric_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for numeric features.
    
    Steps:
    1. Impute missing values with median (robust to outliers)
    2. Scale to zero mean, unit variance
    
    Why StandardScaler?
    - Logistic regression benefits from scaled features
    - Tree-based models don't require it but aren't hurt by it
    - Consistent preprocessing regardless of model choice
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])


def create_binary_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for binary features.
    
    Steps:
    1. Impute missing with most frequent value
    2. Encode Yes/No as 1/0
    
    Note: We don't scale binary features - they're already 0/1.
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', FunctionTransformer(
            encode_binary_feature, 
            validate=False,
            feature_names_out='one-to-one'  # Preserve feature names
        ))
    ])


def create_categorical_pipeline() -> Pipeline:
    """
    Create preprocessing pipeline for categorical features.
    
    Steps:
    1. Impute missing with most frequent value
    2. One-hot encode (creates one column per category)
    
    OneHotEncoder settings:
    - drop='if_binary': For 2-class categories, keeps only 1 column
    - handle_unknown='ignore': Unknown categories become all-zeros
    - sparse_output=False: Returns dense array (easier to debug)
    """
    return Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OneHotEncoder(
            drop='if_binary',
            handle_unknown='ignore',
            sparse_output=False
        ))
    ])


def create_preprocessor() -> ColumnTransformer:
    """
    Create the full preprocessing pipeline using ColumnTransformer.
    
    ColumnTransformer applies different transformations to different
    column groups, then concatenates the results.
    
    Returns:
        Unfitted ColumnTransformer ready to be fitted on training data.
        
    Usage:
        preprocessor = create_preprocessor()
        preprocessor.fit(X_train)  # Fit ONLY on training data
        X_train_transformed = preprocessor.transform(X_train)
        X_val_transformed = preprocessor.transform(X_val)  # Use same fitted preprocessor
    """
    return ColumnTransformer(
        transformers=[
            ('numeric', create_numeric_pipeline(), NUMERIC_COLS),
            ('binary', create_binary_pipeline(), BINARY_COLS),
            ('categorical', create_categorical_pipeline(), CATEGORICAL_COLS)
        ],
        remainder='drop',  # Drop any columns not specified (e.g., customerID)
        verbose_feature_names_out=True  # Prefix feature names with transformer name
    )


def get_feature_names(fitted_preprocessor: ColumnTransformer) -> list:
    """
    Get feature names from a fitted ColumnTransformer.
    
    Args:
        fitted_preprocessor: A ColumnTransformer that has been fitted.
        
    Returns:
        List of feature names in the transformed output.
    """
    return list(fitted_preprocessor.get_feature_names_out())


def preprocess_data(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame
) -> tuple:
    """
    Preprocess train, validation, and test sets.
    
    This is a convenience function that:
    1. Creates a preprocessor
    2. Fits it on training data only
    3. Transforms all three sets
    
    Args:
        X_train: Training features (DataFrame).
        X_val: Validation features (DataFrame).
        X_test: Test features (DataFrame).
        
    Returns:
        Tuple of (X_train_processed, X_val_processed, X_test_processed, preprocessor)
        
    Important:
        The preprocessor is fitted ONLY on X_train.
        This prevents data leakage from validation/test sets.
    """
    preprocessor = create_preprocessor()
    
    # Fit on training data only
    X_train_processed = preprocessor.fit_transform(X_train)
    
    # Transform validation and test using the fitted preprocessor
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)
    
    return X_train_processed, X_val_processed, X_test_processed, preprocessor
