"""
Data splitting utilities.

This module provides functions for creating train/validation/test splits
with proper stratification and reproducibility.
"""

import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split


def create_splits(
    df: pd.DataFrame,
    target_col: str = 'Churn',
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
    test_ratio: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/validation/test splits.
    
    Stratification ensures each split has approximately the same
    proportion of churned customers as the original dataset.
    
    Args:
        df: Full dataset.
        target_col: Name of the target column for stratification.
        train_ratio: Proportion for training set (default 0.6).
        val_ratio: Proportion for validation set (default 0.2).
        test_ratio: Proportion for test set (default 0.2).
        random_state: Random seed for reproducibility.
        
    Returns:
        Tuple of (train_df, val_df, test_df).
        
    Raises:
        ValueError: If ratios don't sum to 1.0.
    """
    # Validate ratios
    total_ratio = train_ratio + val_ratio + test_ratio
    if not np.isclose(total_ratio, 1.0):
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")
    
    # First split: separate test set
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=df[target_col],
        random_state=random_state
    )
    
    # Second split: separate train and validation
    # Adjust validation ratio for the remaining data
    val_ratio_adjusted = val_ratio / (train_ratio + val_ratio)
    
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_adjusted,
        stratify=train_val_df[target_col],
        random_state=random_state
    )
    
    return train_df, val_df, test_df


def get_split_summary(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target_col: str = 'Churn'
) -> dict:
    """
    Generate a summary of the data splits.
    
    Args:
        train_df: Training set.
        val_df: Validation set.
        test_df: Test set.
        target_col: Name of target column.
        
    Returns:
        Dictionary with split statistics.
    """
    def get_stats(df, name):
        churn_rate = df[target_col].mean() * 100
        return {
            'name': name,
            'size': len(df),
            'churn_count': df[target_col].sum(),
            'churn_rate': churn_rate
        }
    
    total = len(train_df) + len(val_df) + len(test_df)
    
    return {
        'total_samples': total,
        'splits': [
            get_stats(train_df, 'train'),
            get_stats(val_df, 'validation'),
            get_stats(test_df, 'test')
        ]
    }


def print_split_summary(summary: dict) -> None:
    """Print a formatted summary of data splits."""
    print("=" * 50)
    print("DATA SPLIT SUMMARY")
    print("=" * 50)
    print(f"Total samples: {summary['total_samples']:,}")
    print()
    print(f"{'Split':<12} {'Size':>8} {'Churn':>8} {'Rate':>8}")
    print("-" * 40)
    
    for split in summary['splits']:
        pct = split['size'] / summary['total_samples'] * 100
        print(
            f"{split['name']:<12} "
            f"{split['size']:>8,} "
            f"{split['churn_count']:>8,} "
            f"{split['churn_rate']:>7.1f}%"
        )
