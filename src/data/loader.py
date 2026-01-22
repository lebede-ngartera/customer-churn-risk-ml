"""
Data loading and validation utilities.

This module provides functions to load the raw dataset and perform
basic validation checks before preprocessing.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Optional


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load the raw Telco Customer Churn dataset.
    
    Args:
        filepath: Path to the CSV file.
        
    Returns:
        Raw DataFrame as loaded from CSV.
        
    Raises:
        FileNotFoundError: If the file doesn't exist.
        ValueError: If the file is empty or missing required columns.
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")
    
    df = pd.read_csv(filepath)
    
    if len(df) == 0:
        raise ValueError("Data file is empty")
    
    # Validate required columns exist
    required_columns = ['customerID', 'Churn', 'TotalCharges', 'tenure']
    missing = set(required_columns) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


def clean_total_charges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the TotalCharges column.
    
    The raw data contains blank strings for customers with tenure=0.
    These are converted to 0.0 (no charges yet for new customers).
    
    Args:
        df: DataFrame with TotalCharges column.
        
    Returns:
        DataFrame with TotalCharges as float64.
        
    Note:
        This operation is performed before train/test split because:
        1. It's a data cleaning step, not a learned transformation
        2. The imputation rule (blank → 0) is domain knowledge, not statistics
        3. No information from other rows is used
    """
    df = df.copy()
    
    # Replace blank strings with NaN, then fill with 0
    df['TotalCharges'] = df['TotalCharges'].replace(r'^\s*$', np.nan, regex=True)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    
    # Fill NaN with 0 (these are tenure=0 customers)
    # Verify assumption: all NaN should have tenure=0
    nan_mask = df['TotalCharges'].isna()
    if nan_mask.any():
        non_zero_tenure_with_nan = df.loc[nan_mask, 'tenure'].sum()
        if non_zero_tenure_with_nan > 0:
            raise ValueError(
                "Found NaN TotalCharges for customers with tenure > 0. "
                "This violates our assumption and requires investigation."
            )
        df['TotalCharges'] = df['TotalCharges'].fillna(0.0)
    
    return df


def encode_target(df: pd.DataFrame, target_col: str = 'Churn') -> pd.DataFrame:
    """
    Encode the target variable as binary (0/1).
    
    Args:
        df: DataFrame with target column.
        target_col: Name of the target column.
        
    Returns:
        DataFrame with binary target.
    """
    df = df.copy()
    df[target_col] = (df[target_col] == 'Yes').astype(int)
    return df


def validate_no_leakage(
    train_ids: pd.Series, 
    val_ids: pd.Series, 
    test_ids: pd.Series
) -> None:
    """
    Verify no customer appears in multiple splits.
    
    Args:
        train_ids: Customer IDs in training set.
        val_ids: Customer IDs in validation set.
        test_ids: Customer IDs in test set.
        
    Raises:
        ValueError: If any customer appears in multiple splits.
    """
    train_set = set(train_ids)
    val_set = set(val_ids)
    test_set = set(test_ids)
    
    train_val_overlap = train_set & val_set
    train_test_overlap = train_set & test_set
    val_test_overlap = val_set & test_set
    
    if train_val_overlap:
        raise ValueError(f"Train-validation overlap: {len(train_val_overlap)} customers")
    if train_test_overlap:
        raise ValueError(f"Train-test overlap: {len(train_test_overlap)} customers")
    if val_test_overlap:
        raise ValueError(f"Validation-test overlap: {len(val_test_overlap)} customers")


def prepare_data(filepath: str) -> pd.DataFrame:
    """
    Load and clean the raw data.
    
    This is the main entry point for data preparation.
    Performs loading, cleaning, and target encoding.
    
    Args:
        filepath: Path to raw CSV file.
        
    Returns:
        Cleaned DataFrame ready for splitting and preprocessing.
    """
    df = load_raw_data(filepath)
    df = clean_total_charges(df)
    df = encode_target(df)
    return df
