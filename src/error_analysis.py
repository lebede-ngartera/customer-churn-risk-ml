"""
Error analysis script for understanding model mistakes.

This script examines patterns in false positives and false negatives
to identify systematic weaknesses and improvement opportunities.

Usage:
    python src/error_analysis.py
    python src/error_analysis.py --threshold 0.5

Outputs:
    - reports/error_analysis.md
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Tuple

import joblib
import numpy as np
import pandas as pd

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


def analyze_errors(
    X: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray
) -> Dict:
    """
    Analyze patterns in prediction errors.
    
    Args:
        X: Feature DataFrame.
        y_true: Ground truth labels.
        y_pred: Predicted labels.
        y_prob: Predicted probabilities.
        
    Returns:
        Dictionary with error analysis results.
    """
    # Create masks for each error type
    true_positives = (y_pred == 1) & (y_true == 1)
    true_negatives = (y_pred == 0) & (y_true == 0)
    false_positives = (y_pred == 1) & (y_true == 0)
    false_negatives = (y_pred == 0) & (y_true == 1)
    
    results = {
        'counts': {
            'true_positives': true_positives.sum(),
            'true_negatives': true_negatives.sum(),
            'false_positives': false_positives.sum(),
            'false_negatives': false_negatives.sum(),
            'total': len(y_true)
        }
    }
    
    # Analyze false negatives (missed churners)
    if false_negatives.sum() > 0:
        fn_data = X[false_negatives]
        churner_data = X[y_true == 1]
        
        results['false_negatives'] = {
            'avg_tenure': fn_data['tenure'].mean(),
            'churner_avg_tenure': churner_data['tenure'].mean(),
            'two_year_contract_pct': (fn_data['Contract'] == 'Two year').mean() if 'Contract' in fn_data.columns else None,
            'churner_two_year_pct': (churner_data['Contract'] == 'Two year').mean() if 'Contract' in churner_data.columns else None,
            'fiber_pct': (fn_data['InternetService'] == 'Fiber optic').mean() if 'InternetService' in fn_data.columns else None,
            'churner_fiber_pct': (churner_data['InternetService'] == 'Fiber optic').mean() if 'InternetService' in churner_data.columns else None,
        }
    
    # Analyze false positives (false alarms)
    if false_positives.sum() > 0:
        fp_data = X[false_positives]
        non_churner_data = X[y_true == 0]
        
        results['false_positives'] = {
            'avg_tenure': fp_data['tenure'].mean(),
            'non_churner_avg_tenure': non_churner_data['tenure'].mean(),
            'month_to_month_pct': (fp_data['Contract'] == 'Month-to-month').mean() if 'Contract' in fp_data.columns else None,
            'non_churner_mtm_pct': (non_churner_data['Contract'] == 'Month-to-month').mean() if 'Contract' in non_churner_data.columns else None,
            'fiber_pct': (fp_data['InternetService'] == 'Fiber optic').mean() if 'InternetService' in fp_data.columns else None,
            'non_churner_fiber_pct': (non_churner_data['InternetService'] == 'Fiber optic').mean() if 'InternetService' in non_churner_data.columns else None,
        }
    
    return results


def generate_error_report(
    analysis: Dict,
    output_path: str
) -> None:
    """Generate error analysis markdown report."""
    
    counts = analysis['counts']
    total = counts['total']
    
    report = f"""# Error Analysis Report

## Overview

This report examines patterns in the model's errors to identify systematic weaknesses.

## Error Distribution (Test Set)

| Error Type | Count | Percentage |
|------------|-------|------------|
| True Positives | {counts['true_positives']} | {100*counts['true_positives']/total:.1f}% |
| True Negatives | {counts['true_negatives']} | {100*counts['true_negatives']/total:.1f}% |
| False Positives | {counts['false_positives']} | {100*counts['false_positives']/total:.1f}% |
| False Negatives | {counts['false_negatives']} | {100*counts['false_negatives']/total:.1f}% |

"""
    
    if 'false_negatives' in analysis:
        fn = analysis['false_negatives']
        report += f"""## False Negative Analysis (Missed Churners)

| Feature | False Negatives | Overall Churners |
|---------|-----------------|------------------|
| Avg tenure (months) | {fn['avg_tenure']:.1f} | {fn['churner_avg_tenure']:.1f} |
| Contract: Two year | {100*fn['two_year_contract_pct']:.0f}% | {100*fn['churner_two_year_pct']:.0f}% |
| Internet: Fiber optic | {100*fn['fiber_pct']:.0f}% | {100*fn['churner_fiber_pct']:.0f}% |

### Insights

1. **Long-tenure customers**: The model underestimates churn risk for established customers.
2. **Two-year contracts**: Customers on long contracts who churn may do so when contracts expire.

"""
    
    if 'false_positives' in analysis:
        fp = analysis['false_positives']
        report += f"""## False Positive Analysis (False Alarms)

| Feature | False Positives | Overall Non-Churners |
|---------|-----------------|----------------------|
| Avg tenure (months) | {fp['avg_tenure']:.1f} | {fp['non_churner_avg_tenure']:.1f} |
| Contract: Month-to-month | {100*fp['month_to_month_pct']:.0f}% | {100*fp['non_churner_mtm_pct']:.0f}% |
| Internet: Fiber optic | {100*fp['fiber_pct']:.0f}% | {100*fp['non_churner_fiber_pct']:.0f}% |

### Insights

1. **New customers**: Month-to-month customers are flagged as high-risk but many stay.
2. **Fiber customers**: Premium service customers may be more loyal than predicted.

"""
    
    report += """## Recommendations

1. **Feature engineering**: Add contract end-date proximity and payment consistency.
2. **Segmented models**: Consider separate models for new vs. established customers.
3. **Threshold adjustment**: Use lower threshold for high-tenure customers.
"""
    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        f.write(report)
    
    logger.info(f"Saved error analysis to {output_path}")


def main(
    model_path: str = 'artifacts/model.joblib',
    data_path: str = 'data/raw/telco_customer_churn.csv',
    threshold: float = 0.5,
    output_dir: str = 'reports'
) -> None:
    """
    Run error analysis and generate report.
    
    Args:
        model_path: Path to fitted model.
        data_path: Path to raw data.
        threshold: Classification threshold.
        output_dir: Directory for output reports.
    """
    print("="*60)
    print(" Error Analysis - Phase 9")
    print("="*60)
    print(f"\nThreshold: {threshold}")
    
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
    y_pred = (y_prob >= threshold).astype(int)
    
    # Run error analysis
    logger.info("Analyzing errors")
    analysis = analyze_errors(X_test, y_test.values, y_pred, y_prob)
    
    # Print summary
    counts = analysis['counts']
    print(f"\nError Distribution:")
    print(f"  True Positives:  {counts['true_positives']}")
    print(f"  True Negatives:  {counts['true_negatives']}")
    print(f"  False Positives: {counts['false_positives']}")
    print(f"  False Negatives: {counts['false_negatives']}")
    
    # Generate report
    generate_error_report(analysis, f"{output_dir}/error_analysis.md")
    
    print(f"\n✓ Report saved to {output_dir}/error_analysis.md")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run error analysis')
    parser.add_argument('--model-path', default='artifacts/model.joblib')
    parser.add_argument('--data-path', default='data/raw/telco_customer_churn.csv')
    parser.add_argument('--threshold', type=float, default=0.5)
    parser.add_argument('--output-dir', default='reports')
    
    args = parser.parse_args()
    
    try:
        main(
            model_path=args.model_path,
            data_path=args.data_path,
            threshold=args.threshold,
            output_dir=args.output_dir
        )
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
