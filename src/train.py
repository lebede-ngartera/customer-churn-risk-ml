"""
Training script for customer churn prediction baseline.

This script trains a Logistic Regression model end-to-end and saves
all artifacts needed for deployment.

Usage:
    python src/train.py
    python src/train.py --data-path data/raw/telco_customer_churn.csv
    python src/train.py --output-dir artifacts

Outputs (saved to artifacts/):
    - model.joblib: Full sklearn Pipeline (preprocessor + classifier)
    - metrics.json: Validation and test set metrics
    - schema.json: Input feature schema for API

Design decisions:
    
1. Why Logistic Regression as baseline?
   - Fast to train, easy to interpret
   - Coefficients directly show feature importance
   - Good baseline before trying complex models
   - Regularization prevents overfitting

2. Why 70/15/15 split?
   - 70% train: Enough data to learn patterns
   - 15% validation: For hyperparameter tuning (future phases)
   - 15% test: Final unbiased evaluation (only used once!)

3. Why stratified split?
   - Churn is imbalanced (~26% positive class)
   - Stratification ensures same class ratio in all splits
   - Prevents unlucky splits with very few positives

4. Why class_weight='balanced'?
   - Churn is imbalanced (26% vs 74%)
   - Without weighting, model might predict "no churn" always
   - 'balanced' automatically adjusts weights inversely to class frequency
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Tuple, Dict, Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import prepare_data, get_feature_names, get_input_schema
from src.evaluate import evaluate_model, print_evaluation_report, save_metrics, compare_results

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# =============================================================================
# Configuration
# =============================================================================

# Random seed for reproducibility
# Using 42 as it's conventional and makes results reproducible
RANDOM_STATE = 42

# Train/validation/test split ratios
# 70% train, 15% val, 15% test
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Logistic Regression hyperparameters
# These are sensible defaults for a baseline model
MODEL_CONFIG = {
    'solver': 'lbfgs',      # Good default, works with L2 penalty
    'max_iter': 1000,       # Increase from default 100 to ensure convergence
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced',  # Handle class imbalance automatically
    'C': 1.0,               # Regularization strength (1/lambda). Lower = more regularization
    'n_jobs': -1            # Use all CPU cores
}


# =============================================================================
# Data Splitting
# =============================================================================

def create_splits(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
    val_ratio: float = VAL_RATIO,
    test_ratio: float = TEST_RATIO,
    random_state: int = RANDOM_STATE
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, 
           pd.Series, pd.Series, pd.Series]:
    """
    Create stratified train/validation/test splits.
    
    Stratification ensures each split has the same proportion of churners
    as the original dataset. This is crucial for imbalanced classification.
    
    Args:
        X: Feature DataFrame.
        y: Target Series.
        train_ratio: Fraction for training (default 0.70).
        val_ratio: Fraction for validation (default 0.15).
        test_ratio: Fraction for testing (default 0.15).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (X_train, X_val, X_test, y_train, y_val, y_test).
    """
    # Validate ratios sum to 1
    total = train_ratio + val_ratio + test_ratio
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total}")
    
    # First split: separate test set
    # test_size is the fraction of remaining data for test
    test_size = test_ratio
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )
    
    # Second split: separate train and validation from remaining
    # val_size is relative to remaining data, not original
    val_size_adjusted = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_size_adjusted,
        stratify=y_temp,
        random_state=random_state
    )
    
    # Log split sizes
    logger.info(f"Data split:")
    logger.info(f"  Train: {len(X_train):,} samples ({len(X_train)/len(X):.1%}), "
                f"churn rate: {y_train.mean():.1%}")
    logger.info(f"  Val:   {len(X_val):,} samples ({len(X_val)/len(X):.1%}), "
                f"churn rate: {y_val.mean():.1%}")
    logger.info(f"  Test:  {len(X_test):,} samples ({len(X_test)/len(X):.1%}), "
                f"churn rate: {y_test.mean():.1%}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test


# =============================================================================
# Model Training
# =============================================================================

def create_model(config: Dict[str, Any] = MODEL_CONFIG) -> LogisticRegression:
    """
    Create a Logistic Regression model with specified configuration.
    
    Logistic Regression hyperparameters explained:
    
    - solver='lbfgs': Limited-memory BFGS, good for small-medium datasets.
      Works with L2 penalty (default). For L1, use 'saga'.
      
    - max_iter=1000: Maximum iterations for convergence. Default 100 often
      isn't enough for one-hot encoded data with many features.
      
    - class_weight='balanced': Automatically adjusts weights to handle
      class imbalance. Weight = n_samples / (n_classes * n_samples_per_class).
      
    - C=1.0: Inverse regularization strength. Lower C = more regularization.
      C=1.0 is sklearn default, provides moderate regularization.
      
    - random_state: Seed for reproducibility (affects tie-breaking).
    
    Args:
        config: Dictionary of hyperparameters.
        
    Returns:
        Unfitted LogisticRegression model.
    """
    logger.info(f"Creating LogisticRegression with config: {config}")
    return LogisticRegression(**config)


def build_pipeline(preprocessor, model) -> Pipeline:
    """
    Build a sklearn Pipeline combining preprocessing and model.
    
    Why use a Pipeline?
    1. Prevents data leakage: preprocessor is fitted only on training data
    2. Simplifies deployment: single object to serialize and load
    3. Cleaner code: one fit() and predict() call handles everything
    
    Args:
        preprocessor: Unfitted ColumnTransformer.
        model: Unfitted classifier.
        
    Returns:
        Unfitted Pipeline ready for training.
    """
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
    
    logger.info("Built pipeline: preprocessor -> classifier")
    return pipeline


def train_pipeline(
    pipeline: Pipeline,
    X_train: pd.DataFrame,
    y_train: pd.Series
) -> Pipeline:
    """
    Fit the pipeline on training data.
    
    Args:
        pipeline: Unfitted Pipeline.
        X_train: Training features.
        y_train: Training labels.
        
    Returns:
        Fitted Pipeline.
    """
    logger.info(f"Training pipeline on {len(X_train):,} samples...")
    pipeline.fit(X_train, y_train)
    logger.info("Training complete")
    
    return pipeline


# =============================================================================
# Artifact Saving
# =============================================================================

def save_artifacts(
    pipeline: Pipeline,
    val_results: Dict[str, Any],
    test_results: Dict[str, Any],
    output_dir: str = 'artifacts'
) -> None:
    """
    Save all training artifacts to disk.
    
    Artifacts saved:
    1. model.joblib - Full fitted pipeline (for inference)
    2. metrics.json - Validation and test metrics (for monitoring)
    3. schema.json - Input feature schema (for API validation)
    
    Args:
        pipeline: Fitted sklearn Pipeline.
        val_results: Validation set evaluation results.
        test_results: Test set evaluation results.
        output_dir: Directory to save artifacts.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Save model pipeline
    model_path = output_path / 'model.joblib'
    joblib.dump(pipeline, model_path)
    logger.info(f"Saved model to {model_path}")
    
    # 2. Save metrics
    metrics_data = {
        'training_date': datetime.now().isoformat(),
        'model_type': 'LogisticRegression',
        'model_config': MODEL_CONFIG,
        'validation': val_results,
        'test': test_results
    }
    metrics_path = output_path / 'metrics.json'
    save_metrics(metrics_data, str(metrics_path))
    
    # 3. Save input schema
    schema = get_input_schema()
    # Add transformed feature names (from fitted preprocessor)
    preprocessor = pipeline.named_steps['preprocessor']
    schema['transformed_features'] = get_feature_names(preprocessor)
    schema['n_transformed_features'] = len(schema['transformed_features'])
    
    schema_path = output_path / 'schema.json'
    with open(schema_path, 'w') as f:
        json.dump(schema, f, indent=2)
    logger.info(f"Saved schema to {schema_path}")
    
    # Print summary
    print(f"\n{'='*60}")
    print(" Artifacts Saved")
    print(f"{'='*60}")
    print(f"  Model:   {model_path}")
    print(f"  Metrics: {metrics_path}")
    print(f"  Schema:  {schema_path}")


# =============================================================================
# Main Training Function
# =============================================================================

def main(
    data_path: str = 'data/raw/telco_customer_churn.csv',
    output_dir: str = 'artifacts'
) -> None:
    """
    Run the full training pipeline.
    
    This is the main entry point that:
    1. Loads and preprocesses data
    2. Creates train/val/test splits
    3. Trains a Logistic Regression pipeline
    4. Evaluates on validation and test sets
    5. Saves all artifacts
    
    Args:
        data_path: Path to raw data CSV.
        output_dir: Directory for saving artifacts.
    """
    print("="*60)
    print(" Customer Churn Prediction - Baseline Training")
    print("="*60)
    print(f"\nData path:   {data_path}")
    print(f"Output dir:  {output_dir}")
    print(f"Random seed: {RANDOM_STATE}")
    
    # Step 1: Load and prepare data
    print("\n" + "-"*40)
    print("Step 1: Loading and preparing data")
    print("-"*40)
    X, y, preprocessor = prepare_data(data_path)
    
    # Step 2: Create splits
    print("\n" + "-"*40)
    print("Step 2: Creating train/val/test splits")
    print("-"*40)
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)
    
    # Step 3: Build and train model
    print("\n" + "-"*40)
    print("Step 3: Training model")
    print("-"*40)
    model = create_model()
    pipeline = build_pipeline(preprocessor, model)
    pipeline = train_pipeline(pipeline, X_train, y_train)
    
    # Step 4: Evaluate on validation set
    print("\n" + "-"*40)
    print("Step 4: Evaluating on validation set")
    print("-"*40)
    val_results = evaluate_model(pipeline, X_val, y_val, 'validation')
    print_evaluation_report(val_results, "Validation Set Results")
    
    # Step 5: Evaluate on test set (final, unbiased evaluation)
    print("\n" + "-"*40)
    print("Step 5: Evaluating on test set")
    print("-"*40)
    test_results = evaluate_model(pipeline, X_test, y_test, 'test')
    print_evaluation_report(test_results, "Test Set Results (Final)")
    
    # Compare validation and test results
    compare_results(val_results, test_results)
    
    # Step 6: Save artifacts
    print("\n" + "-"*40)
    print("Step 6: Saving artifacts")
    print("-"*40)
    save_artifacts(pipeline, val_results, test_results, output_dir)
    
    # Final summary
    print("\n" + "="*60)
    print(" Training Complete!")
    print("="*60)
    test_auc = test_results['metrics'].get('roc_auc', 'N/A')
    print(f"\nFinal Test ROC-AUC: {test_auc:.4f}" if isinstance(test_auc, float) else f"\nFinal Test ROC-AUC: {test_auc}")
    print(f"\nTo use the trained model:")
    print(f"  import joblib")
    print(f"  pipeline = joblib.load('{output_dir}/model.joblib')")
    print(f"  predictions = pipeline.predict(X_new)")
    print(f"  probabilities = pipeline.predict_proba(X_new)[:, 1]")


if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description='Train customer churn prediction baseline model'
    )
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/raw/telco_customer_churn.csv',
        help='Path to raw data CSV (default: data/raw/telco_customer_churn.csv)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='artifacts',
        help='Directory for saving artifacts (default: artifacts)'
    )
    
    args = parser.parse_args()
    
    try:
        main(data_path=args.data_path, output_dir=args.output_dir)
    except FileNotFoundError as e:
        logger.error(f"Data file not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
