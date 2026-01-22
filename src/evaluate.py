"""
Evaluation module for customer churn prediction.

This module provides functions to compute classification metrics
and generate readable evaluation reports.

Metric choices explained:
- ROC-AUC: Primary metric. Measures ranking ability across all thresholds.
  Good for imbalanced data because it's threshold-independent.
  
- Precision: Of customers we predict will churn, what fraction actually do?
  High precision = fewer false alarms (wasted retention offers).
  
- Recall: Of customers who actually churn, what fraction do we catch?
  High recall = fewer missed churners (lost customers).
  
- F1: Harmonic mean of precision and recall. Useful single summary,
  but hides the precision/recall tradeoff.

Business context:
- Cost of false positive (predict churn, customer stays): Wasted retention offer (~$50)
- Cost of false negative (predict stay, customer churns): Lost customer (~$500+ CLV)
- This asymmetry suggests we should favor recall over precision.
"""

import json
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Compute classification metrics for binary churn prediction.
    
    Args:
        y_true: Ground truth labels (0 or 1).
        y_pred: Predicted labels (0 or 1).
        y_prob: Predicted probabilities for class 1 (optional, needed for ROC-AUC).
        
    Returns:
        Dictionary of metric names to values.
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0)
    }
    
    # ROC-AUC requires probability scores
    if y_prob is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
    
    return metrics


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Compute confusion matrix and return as labeled dictionary.
    
    Matrix layout:
                     Predicted
                   No      Yes
    Actual  No    TN       FP
            Yes   FN       TP
    
    Args:
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary with TN, FP, FN, TP counts.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    return {
        'true_negatives': int(tn),   # Correctly predicted no churn
        'false_positives': int(fp),  # Predicted churn, actually stayed
        'false_negatives': int(fn),  # Predicted stay, actually churned
        'true_positives': int(tp)    # Correctly predicted churn
    }


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    dataset_name: str = 'test'
) -> Dict[str, Any]:
    """
    Evaluate a fitted model on a dataset.
    
    Args:
        model: Fitted sklearn model/pipeline with predict and predict_proba.
        X: Feature matrix.
        y: True labels.
        dataset_name: Name for logging (e.g., 'validation', 'test').
        
    Returns:
        Dictionary containing metrics and confusion matrix.
    """
    logger.info(f"Evaluating model on {dataset_name} set ({len(y)} samples)")
    
    # Get predictions
    y_pred = model.predict(X)
    
    # Get probabilities if available
    y_prob = None
    if hasattr(model, 'predict_proba'):
        y_prob = model.predict_proba(X)[:, 1]  # Probability of class 1 (churn)
    
    # Compute metrics
    metrics = compute_metrics(y, y_pred, y_prob)
    conf_matrix = compute_confusion_matrix(y, y_pred)
    
    return {
        'dataset': dataset_name,
        'n_samples': len(y),
        'metrics': metrics,
        'confusion_matrix': conf_matrix
    }


def print_evaluation_report(
    results: Dict[str, Any],
    title: Optional[str] = None
) -> None:
    """
    Print a formatted evaluation report to console.
    
    Args:
        results: Output from evaluate_model().
        title: Optional title for the report.
    """
    if title:
        print(f"\n{'='*60}")
        print(f" {title}")
        print(f"{'='*60}")
    
    dataset = results.get('dataset', 'unknown')
    n_samples = results.get('n_samples', 0)
    metrics = results.get('metrics', {})
    cm = results.get('confusion_matrix', {})
    
    print(f"\nDataset: {dataset} ({n_samples:,} samples)")
    print("-" * 40)
    
    # Metrics
    print("\nClassification Metrics:")
    print(f"  ROC-AUC:   {metrics.get('roc_auc', 'N/A'):>7.4f}" if 'roc_auc' in metrics else "  ROC-AUC:   N/A")
    print(f"  Accuracy:  {metrics.get('accuracy', 0):>7.4f}")
    print(f"  Precision: {metrics.get('precision', 0):>7.4f}")
    print(f"  Recall:    {metrics.get('recall', 0):>7.4f}")
    print(f"  F1 Score:  {metrics.get('f1', 0):>7.4f}")
    
    # Confusion Matrix
    print("\nConfusion Matrix:")
    print("                  Predicted")
    print("                 No     Yes")
    print(f"  Actual No   {cm.get('true_negatives', 0):>5}   {cm.get('false_positives', 0):>5}")
    print(f"  Actual Yes  {cm.get('false_negatives', 0):>5}   {cm.get('true_positives', 0):>5}")
    
    # Business interpretation
    total = sum(cm.values()) if cm else 1
    fp_rate = cm.get('false_positives', 0) / total
    fn_rate = cm.get('false_negatives', 0) / total
    
    print("\nBusiness Impact:")
    print(f"  False positives: {cm.get('false_positives', 0):,} "
          f"({fp_rate:.1%}) - unnecessary retention offers")
    print(f"  False negatives: {cm.get('false_negatives', 0):,} "
          f"({fn_rate:.1%}) - missed churners (lost customers)")


def save_metrics(
    results: Dict[str, Any],
    filepath: str
) -> None:
    """
    Save evaluation results to a JSON file.
    
    Args:
        results: Evaluation results dictionary.
        filepath: Output path for JSON file.
    """
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        return obj
    
    results_clean = convert_numpy(results)
    
    with open(path, 'w') as f:
        json.dump(results_clean, f, indent=2)
    
    logger.info(f"Saved metrics to {filepath}")


def compare_results(
    val_results: Dict[str, Any],
    test_results: Dict[str, Any]
) -> None:
    """
    Compare validation and test results to check for overfitting.
    
    A large gap between validation and test performance may indicate:
    - Overfitting to the validation set during hyperparameter tuning
    - Data distribution shift between sets
    
    Args:
        val_results: Evaluation results on validation set.
        test_results: Evaluation results on test set.
    """
    print("\n" + "="*60)
    print(" Validation vs Test Comparison")
    print("="*60)
    
    val_metrics = val_results.get('metrics', {})
    test_metrics = test_results.get('metrics', {})
    
    print("\nMetric         Validation    Test      Gap")
    print("-" * 50)
    
    for metric in ['roc_auc', 'accuracy', 'precision', 'recall', 'f1']:
        val_val = val_metrics.get(metric)
        test_val = test_metrics.get(metric)
        
        if val_val is not None and test_val is not None:
            gap = val_val - test_val
            gap_str = f"{gap:+.4f}"
            status = "⚠️" if abs(gap) > 0.05 else "✓"
            print(f"{metric:<12}   {val_val:.4f}       {test_val:.4f}    {gap_str} {status}")
    
    print("\n(Gap > 0.05 may indicate overfitting)")


if __name__ == '__main__':
    # Quick test with synthetic data
    print("Testing evaluation module with synthetic data...")
    
    np.random.seed(42)
    n = 100
    y_true = np.random.randint(0, 2, n)
    y_pred = np.random.randint(0, 2, n)
    y_prob = np.random.rand(n)
    
    metrics = compute_metrics(y_true, y_pred, y_prob)
    print(f"\nMetrics: {metrics}")
    
    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"Confusion matrix: {cm}")
    
    print("\n✓ Evaluation module working correctly")
