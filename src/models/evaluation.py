"""
Model evaluation utilities.

This module provides functions for evaluating model performance
with appropriate metrics for churn prediction.

Metric Selection Rationale:
- ROC-AUC: Primary metric. Threshold-independent, good for ranking.
- Precision: Of predicted churns, how many actually churned?
- Recall: Of actual churns, how many did we catch?
- F1: Harmonic mean of precision and recall.

Business Context:
- False Negative (missed churn): Lost customer, high cost
- False Positive (false alarm): Unnecessary retention effort, low cost
- Therefore, we often prefer higher recall (catch more churns)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict[str, float]:
    """
    Calculate classification metrics.
    
    Args:
        y_true: True labels (0/1).
        y_pred: Predicted labels (0/1).
        y_prob: Predicted probabilities for class 1.
        
    Returns:
        Dictionary of metric names to values.
    """
    return {
        'roc_auc': roc_auc_score(y_true, y_prob),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def calculate_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray
) -> Dict[str, int]:
    """
    Calculate confusion matrix components.
    
    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        
    Returns:
        Dictionary with TN, FP, FN, TP counts.
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }


def evaluate_model(
    model,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Comprehensive model evaluation.
    
    Args:
        model: Fitted model with predict_proba method.
        X: Features (preprocessed).
        y: True labels.
        threshold: Classification threshold for probabilities.
        
    Returns:
        Dictionary containing all evaluation results.
    """
    # Get predictions
    y_prob = model.predict_proba(X)[:, 1]
    y_pred = (y_prob >= threshold).astype(int)
    
    # Calculate metrics
    metrics = calculate_metrics(y, y_pred, y_prob)
    cm = calculate_confusion_matrix(y, y_pred)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'threshold': threshold,
        'n_samples': len(y),
        'n_positive': int(y.sum()),
        'n_predicted_positive': int(y_pred.sum())
    }


def compare_models(
    results: Dict[str, Dict[str, Any]]
) -> pd.DataFrame:
    """
    Create a comparison table of model results.
    
    Args:
        results: Dictionary mapping model names to evaluation results.
        
    Returns:
        DataFrame with models as rows and metrics as columns.
    """
    rows = []
    for model_name, result in results.items():
        row = {'model': model_name}
        row.update(result['metrics'])
        rows.append(row)
    
    df = pd.DataFrame(rows)
    df = df.set_index('model')
    
    # Sort by ROC-AUC descending
    df = df.sort_values('roc_auc', ascending=False)
    
    return df


def print_evaluation_report(
    model_name: str,
    results: Dict[str, Any],
    show_confusion_matrix: bool = True
) -> None:
    """
    Print a formatted evaluation report.
    
    Args:
        model_name: Name of the model.
        results: Evaluation results from evaluate_model().
        show_confusion_matrix: Whether to show confusion matrix.
    """
    print("=" * 60)
    print(f"EVALUATION REPORT: {model_name}")
    print("=" * 60)
    
    metrics = results['metrics']
    print(f"\nMetrics (threshold = {results['threshold']}):")
    print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
    print(f"  Precision: {metrics['precision']:.4f}")
    print(f"  Recall:    {metrics['recall']:.4f}")
    print(f"  F1 Score:  {metrics['f1']:.4f}")
    
    if show_confusion_matrix:
        cm = results['confusion_matrix']
        print(f"\nConfusion Matrix:")
        print(f"                    Predicted")
        print(f"                 No Churn  Churn")
        print(f"  Actual No Churn   {cm['true_negatives']:5d}  {cm['false_positives']:5d}")
        print(f"  Actual Churn      {cm['false_negatives']:5d}  {cm['true_positives']:5d}")
    
    print(f"\nSummary:")
    print(f"  Total samples: {results['n_samples']}")
    print(f"  Actual churns: {results['n_positive']}")
    print(f"  Predicted churns: {results['n_predicted_positive']}")


def get_roc_curve_data(
    model,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get ROC curve data for plotting.
    
    Args:
        model: Fitted model.
        X: Features.
        y: True labels.
        
    Returns:
        Tuple of (fpr, tpr, thresholds).
    """
    y_prob = model.predict_proba(X)[:, 1]
    return roc_curve(y, y_prob)


def get_precision_recall_curve_data(
    model,
    X: np.ndarray,
    y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Get precision-recall curve data for plotting.
    
    Args:
        model: Fitted model.
        X: Features.
        y: True labels.
        
    Returns:
        Tuple of (precision, recall, thresholds).
    """
    y_prob = model.predict_proba(X)[:, 1]
    return precision_recall_curve(y, y_prob)


def find_optimal_threshold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    metric: str = 'f1'
) -> Tuple[float, float]:
    """
    Find the optimal classification threshold.
    
    Args:
        model: Fitted model.
        X: Features.
        y: True labels.
        metric: Metric to optimize ('f1', 'recall', 'precision').
        
    Returns:
        Tuple of (optimal_threshold, best_metric_value).
    """
    y_prob = model.predict_proba(X)[:, 1]
    
    best_threshold = 0.5
    best_score = 0.0
    
    for threshold in np.arange(0.1, 0.9, 0.01):
        y_pred = (y_prob >= threshold).astype(int)
        
        if metric == 'f1':
            score = f1_score(y, y_pred, zero_division=0)
        elif metric == 'recall':
            score = recall_score(y, y_pred, zero_division=0)
        elif metric == 'precision':
            score = precision_score(y, y_pred, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    return best_threshold, best_score


def explain_metrics_business_context() -> str:
    """
    Return a plain-English explanation of metrics in business context.
    """
    return """
METRICS EXPLAINED IN BUSINESS CONTEXT
=====================================

ROC-AUC (Area Under ROC Curve):
  What it measures: Overall ability to rank customers by churn risk.
  Range: 0.5 (random) to 1.0 (perfect).
  Business meaning: A higher AUC means the model better separates 
  high-risk from low-risk customers, regardless of threshold.

Precision:
  What it measures: Of customers we predict will churn, what % actually do?
  Example: Precision = 0.65 means 65% of flagged customers actually churn.
  Business meaning: Higher precision = fewer wasted retention efforts.

Recall (Sensitivity):
  What it measures: Of customers who actually churn, what % did we catch?
  Example: Recall = 0.80 means we catch 80% of churners.
  Business meaning: Higher recall = fewer missed churns (lost customers).

F1 Score:
  What it measures: Balance between precision and recall.
  Business meaning: Useful when you want to balance catching churns 
  vs. not wasting resources on false alarms.

THE PRECISION-RECALL TRADEOFF:
  - If you lower the threshold: More customers flagged as churn risk
    → Higher recall (catch more churns)
    → Lower precision (more false alarms)
  
  - If you raise the threshold: Fewer customers flagged
    → Lower recall (miss more churns)
    → Higher precision (fewer false alarms)

WHICH METRIC TO PRIORITIZE?
  It depends on business costs:
  - If losing a customer is very expensive → prioritize RECALL
  - If retention campaigns are expensive → prioritize PRECISION
  - If costs are balanced → prioritize F1 or tune threshold
"""
