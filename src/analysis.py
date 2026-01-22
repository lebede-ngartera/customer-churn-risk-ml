"""
Model analysis script for threshold tuning and calibration assessment.

This script generates reports for Phase 9: Model Review & Analysis.

Usage:
    python src/analysis.py
    python src/analysis.py --model-path artifacts/model.joblib

Outputs:
    - reports/threshold_analysis.md
    - reports/calibration.md
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    f1_score
)

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.preprocessing import prepare_data
from src.train import create_splits, RANDOM_STATE

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def analyze_thresholds(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    thresholds: List[float] = None
) -> pd.DataFrame:
    """
    Analyze precision, recall, and F1 at different thresholds.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        thresholds: List of thresholds to evaluate.
        
    Returns:
        DataFrame with metrics at each threshold.
    """
    if thresholds is None:
        thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
    
    results = []
    total_positives = y_true.sum()
    
    for thresh in thresholds:
        y_pred = (y_prob >= thresh).astype(int)
        
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        
        true_positives = ((y_pred == 1) & (y_true == 1)).sum()
        false_positives = ((y_pred == 1) & (y_true == 0)).sum()
        
        results.append({
            'threshold': thresh,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'churners_caught': f"{true_positives}/{total_positives}",
            'false_alarms': false_positives
        })
    
    return pd.DataFrame(results)


def analyze_calibration(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Tuple[pd.DataFrame, float]:
    """
    Analyze probability calibration.
    
    Args:
        y_true: Ground truth labels.
        y_prob: Predicted probabilities.
        n_bins: Number of calibration bins.
        
    Returns:
        Tuple of (calibration DataFrame, Brier score).
    """
    # Compute calibration curve
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
    
    # Compute Brier score
    brier = brier_score_loss(y_true, y_prob)
    
    # Create binned analysis
    bins = np.linspace(0, 1, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)
    
    results = []
    for i in range(n_bins):
        mask = bin_indices == i
        if mask.sum() > 0:
            actual_rate = y_true[mask].mean()
            predicted_mid = (bins[i] + bins[i+1]) / 2
            gap = actual_rate - predicted_mid
            
            results.append({
                'bin': f"{bins[i]:.1f} - {bins[i+1]:.1f}",
                'actual_churn_rate': actual_rate,
                'count': mask.sum(),
                'gap': gap
            })
    
    return pd.DataFrame(results), brier


def generate_threshold_report(
    threshold_df: pd.DataFrame,
    output_path: str
) -> None:
    """Generate threshold analysis markdown report."""
    
    report = """# Threshold Analysis Report

## Overview

This report analyzes the precision-recall tradeoff at different classification thresholds for the customer churn model.

## Threshold Sensitivity

| Threshold | Precision | Recall | F1 | Churners Caught | False Alarms |
|-----------|-----------|--------|-----|-----------------|--------------|
"""
    
    for _, row in threshold_df.iterrows():
        report += f"| {row['threshold']} | {row['precision']:.2f} | {row['recall']:.2f} | {row['f1']:.2f} | {row['churners_caught']} | {row['false_alarms']} |\n"
    
    report += """
## Recommendations

Given the 10:1 cost asymmetry (false negative = $500, false positive = $50):

- **Conservative threshold (0.3-0.4)**: Catches more churners but generates more false alarms.
- **Balanced threshold (0.5)**: Default sklearn behavior. Reasonable starting point.
- **Aggressive threshold (0.6-0.7)**: Higher precision, fewer wasted offers.

The optimal threshold depends on retention team capacity and intervention costs.
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved threshold analysis to {output_path}")


def generate_calibration_report(
    calibration_df: pd.DataFrame,
    brier_score: float,
    output_path: str
) -> None:
    """Generate calibration markdown report."""
    
    report = f"""# Calibration Report

## Overview

This report assesses whether the model's predicted probabilities match observed churn rates.

## Brier Score: {brier_score:.3f}

Lower is better. A Brier score of 0.25 corresponds to random guessing for balanced classes.

## Calibration by Probability Bucket

| Predicted Probability | Actual Churn Rate | Count | Gap |
|-----------------------|-------------------|-------|-----|
"""
    
    for _, row in calibration_df.iterrows():
        gap_str = f"+{row['gap']:.2f}" if row['gap'] >= 0 else f"{row['gap']:.2f}"
        report += f"| {row['bin']} | {row['actual_churn_rate']:.2f} | {row['count']} | {gap_str} |\n"
    
    report += """
## Observations

The model is sufficiently calibrated for ranking customers by risk.
If exact probability estimates are critical, consider applying isotonic regression calibration.
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved calibration report to {output_path}")


def main(
    model_path: str = 'artifacts/model.joblib',
    data_path: str = 'data/raw/telco_customer_churn.csv',
    output_dir: str = 'reports'
) -> None:
    """
    Run model analysis and generate reports.
    
    Args:
        model_path: Path to fitted model.
        data_path: Path to raw data.
        output_dir: Directory for output reports.
    """
    print("="*60)
    print(" Model Analysis - Phase 9")
    print("="*60)
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    pipeline = joblib.load(model_path)
    
    # Load and split data (same split as training)
    logger.info("Loading and splitting data")
    X, y, _ = prepare_data(data_path)
    X_train, X_val, X_test, y_train, y_val, y_test = create_splits(X, y)
    
    # Get predictions on test set
    logger.info("Generating predictions on test set")
    y_prob = pipeline.predict_proba(X_test)[:, 1]
    
    # Threshold analysis
    logger.info("Analyzing thresholds")
    threshold_df = analyze_thresholds(y_test.values, y_prob)
    print("\nThreshold Analysis:")
    print(threshold_df.to_string(index=False))
    
    # Calibration analysis
    logger.info("Analyzing calibration")
    calibration_df, brier = analyze_calibration(y_test.values, y_prob)
    print(f"\nBrier Score: {brier:.4f}")
    
    # Generate reports
    generate_threshold_report(
        threshold_df,
        f"{output_dir}/threshold_analysis.md"
    )
    generate_calibration_report(
        calibration_df,
        brier,
        f"{output_dir}/calibration.md"
    )
    
    print(f"\n✓ Reports saved to {output_dir}/")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run model analysis')
    parser.add_argument('--model-path', default='artifacts/model.joblib')
    parser.add_argument('--data-path', default='data/raw/telco_customer_churn.csv')
    parser.add_argument('--output-dir', default='reports')
    
    args = parser.parse_args()
    
    try:
        main(
            model_path=args.model_path,
            data_path=args.data_path,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
