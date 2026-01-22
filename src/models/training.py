"""
Model training utilities.

This module provides functions for training and managing models
for customer churn prediction.

Model Selection Rationale:
1. Logistic Regression: Interpretable baseline, fast, works well with scaled features
2. Random Forest: Ensemble of trees, handles non-linearity, resistant to overfitting
3. Gradient Boosting: Sequential ensemble, often best performance, more prone to overfit
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
import joblib
from pathlib import Path


def create_logistic_regression(
    C: float = 1.0,
    max_iter: int = 1000,
    random_state: int = 42
) -> LogisticRegression:
    """
    Create a logistic regression model.
    
    Logistic regression is our baseline because:
    - It's interpretable (coefficients show feature importance direction)
    - It's fast to train and predict
    - It provides well-calibrated probabilities
    - It works well when features are scaled (which we've done)
    
    Args:
        C: Inverse regularization strength. Smaller = stronger regularization.
           Default 1.0 is a reasonable starting point.
        max_iter: Maximum iterations for solver convergence.
        random_state: Random seed for reproducibility.
        
    Returns:
        Unfitted LogisticRegression model.
    
    Bias-Variance Tradeoff:
        - High C (weak regularization): Lower bias, higher variance (may overfit)
        - Low C (strong regularization): Higher bias, lower variance (may underfit)
        - Default C=1.0 is a balanced starting point
    """
    return LogisticRegression(
        C=C,
        max_iter=max_iter,
        random_state=random_state,
        solver='lbfgs',  # Good default for small-medium datasets
        class_weight=None  # We'll handle imbalance through metrics, not weights
    )


def create_random_forest(
    n_estimators: int = 100,
    max_depth: Optional[int] = 10,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> RandomForestClassifier:
    """
    Create a random forest classifier.
    
    Random Forest is included because:
    - It captures non-linear relationships
    - It's resistant to overfitting due to bagging
    - It provides feature importance
    - It handles mixed feature types well
    
    Args:
        n_estimators: Number of trees. More trees = more stable but slower.
        max_depth: Maximum tree depth. None = unlimited (may overfit).
        min_samples_leaf: Minimum samples in leaf nodes. Higher = more regularization.
        random_state: Random seed for reproducibility.
        
    Returns:
        Unfitted RandomForestClassifier.
    
    Bias-Variance Tradeoff:
        - Deep trees (high max_depth): Lower bias, higher variance
        - Shallow trees (low max_depth): Higher bias, lower variance
        - More trees reduce variance without increasing bias
        - We use max_depth=10 as a conservative default to prevent overfitting
    """
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        n_jobs=-1,  # Use all CPU cores
        class_weight=None
    )


def create_gradient_boosting(
    n_estimators: int = 100,
    max_depth: int = 3,
    learning_rate: float = 0.1,
    min_samples_leaf: int = 5,
    random_state: int = 42
) -> GradientBoostingClassifier:
    """
    Create a gradient boosting classifier.
    
    Gradient Boosting is included because:
    - It often achieves best performance on tabular data
    - It builds trees sequentially, correcting previous errors
    - It's highly flexible
    
    Args:
        n_estimators: Number of boosting stages.
        max_depth: Maximum depth of individual trees. Typically 3-5 for boosting.
        learning_rate: Shrinkage parameter. Lower = more conservative, needs more trees.
        min_samples_leaf: Minimum samples in leaf nodes.
        random_state: Random seed for reproducibility.
        
    Returns:
        Unfitted GradientBoostingClassifier.
    
    Bias-Variance Tradeoff:
        - Higher learning_rate + fewer trees: Faster but may overfit
        - Lower learning_rate + more trees: Slower but better generalization
        - Shallow trees (max_depth=3) are standard for boosting
        - Gradient boosting is MORE prone to overfitting than Random Forest
          because trees are correlated (each corrects previous errors)
    """
    return GradientBoostingClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        learning_rate=learning_rate,
        min_samples_leaf=min_samples_leaf,
        random_state=random_state,
        validation_fraction=0.1,  # Use 10% of training for early stopping
        n_iter_no_change=10,  # Stop if no improvement for 10 rounds
        tol=1e-4
    )


def get_model_catalog() -> Dict[str, Dict[str, Any]]:
    """
    Return a catalog of available models with their configurations.
    
    Returns:
        Dictionary mapping model names to their configurations.
    """
    return {
        'logistic_regression': {
            'name': 'Logistic Regression',
            'create_fn': create_logistic_regression,
            'description': 'Linear baseline model. Interpretable coefficients.',
            'training_time': 'Fast (~seconds)',
            'interpretability': 'High',
        },
        'random_forest': {
            'name': 'Random Forest',
            'create_fn': create_random_forest,
            'description': 'Ensemble of decision trees with bagging.',
            'training_time': 'Medium (~minutes)',
            'interpretability': 'Medium (feature importance)',
        },
        'gradient_boosting': {
            'name': 'Gradient Boosting',
            'create_fn': create_gradient_boosting,
            'description': 'Sequential ensemble that corrects errors.',
            'training_time': 'Medium (~minutes)',
            'interpretability': 'Medium (feature importance)',
        }
    }


def train_model(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray
) -> Any:
    """
    Train a model on the training data.
    
    Args:
        model: Unfitted sklearn model.
        X_train: Training features (preprocessed).
        y_train: Training labels.
        
    Returns:
        Fitted model.
    """
    model.fit(X_train, y_train)
    return model


def save_model(
    model,
    filepath: str,
    preprocessor: Optional[ColumnTransformer] = None
) -> str:
    """
    Save a trained model (and optionally preprocessor) to disk.
    
    Args:
        model: Fitted model.
        filepath: Path to save the model.
        preprocessor: Optional fitted preprocessor to bundle with model.
        
    Returns:
        Path where model was saved.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Bundle model and preprocessor together
    artifact = {
        'model': model,
        'preprocessor': preprocessor
    }
    
    joblib.dump(artifact, path)
    return str(path)


def load_model(filepath: str) -> Tuple[Any, Optional[ColumnTransformer]]:
    """
    Load a saved model and preprocessor.
    
    Args:
        filepath: Path to the saved model.
        
    Returns:
        Tuple of (model, preprocessor). Preprocessor may be None.
    """
    artifact = joblib.load(filepath)
    return artifact['model'], artifact.get('preprocessor')
