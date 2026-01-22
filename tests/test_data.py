"""
Tests for data loading and cleaning.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.loader import (
    clean_total_charges,
    encode_target,
    validate_no_leakage
)
from src.data.splitting import create_splits


class TestCleanTotalCharges:
    """Tests for TotalCharges cleaning."""
    
    def test_converts_blank_to_zero(self):
        """Blank strings should become 0.0 when tenure is 0."""
        df = pd.DataFrame({
            'TotalCharges': ['100.5', ' ', '200.0'],
            'tenure': [10, 0, 20]
        })
        result = clean_total_charges(df)
        
        assert result['TotalCharges'].dtype == np.float64
        assert result.loc[1, 'TotalCharges'] == 0.0
    
    def test_preserves_valid_values(self):
        """Valid numeric strings should be converted correctly."""
        df = pd.DataFrame({
            'TotalCharges': ['100.50', '200.75', '0.0'],
            'tenure': [10, 20, 0]
        })
        result = clean_total_charges(df)
        
        assert result.loc[0, 'TotalCharges'] == 100.50
        assert result.loc[1, 'TotalCharges'] == 200.75
        assert result.loc[2, 'TotalCharges'] == 0.0
    
    def test_raises_on_invalid_data(self):
        """Should raise if NaN TotalCharges has non-zero tenure."""
        df = pd.DataFrame({
            'TotalCharges': [' '],  # Blank
            'tenure': [5]  # Non-zero tenure - this shouldn't happen
        })
        
        with pytest.raises(ValueError, match="tenure > 0"):
            clean_total_charges(df)


class TestEncodeTarget:
    """Tests for target encoding."""
    
    def test_encodes_yes_as_one(self):
        """'Yes' should become 1."""
        df = pd.DataFrame({'Churn': ['Yes', 'No', 'Yes']})
        result = encode_target(df)
        
        assert result['Churn'].tolist() == [1, 0, 1]
    
    def test_preserves_other_columns(self):
        """Other columns should not be modified."""
        df = pd.DataFrame({
            'Churn': ['Yes', 'No'],
            'other': ['a', 'b']
        })
        result = encode_target(df)
        
        assert result['other'].tolist() == ['a', 'b']


class TestValidateNoLeakage:
    """Tests for leakage validation."""
    
    def test_passes_with_no_overlap(self):
        """Should pass when no customers overlap."""
        train = pd.Series(['A', 'B', 'C'])
        val = pd.Series(['D', 'E'])
        test = pd.Series(['F', 'G'])
        
        # Should not raise
        validate_no_leakage(train, val, test)
    
    def test_raises_on_train_val_overlap(self):
        """Should raise when train and validation overlap."""
        train = pd.Series(['A', 'B', 'C'])
        val = pd.Series(['C', 'D'])  # 'C' overlaps
        test = pd.Series(['E', 'F'])
        
        with pytest.raises(ValueError, match="Train-validation overlap"):
            validate_no_leakage(train, val, test)
    
    def test_raises_on_train_test_overlap(self):
        """Should raise when train and test overlap."""
        train = pd.Series(['A', 'B', 'C'])
        val = pd.Series(['D', 'E'])
        test = pd.Series(['A', 'F'])  # 'A' overlaps
        
        with pytest.raises(ValueError, match="Train-test overlap"):
            validate_no_leakage(train, val, test)


class TestCreateSplits:
    """Tests for data splitting."""
    
    @pytest.fixture
    def sample_df(self):
        """Create a sample DataFrame for testing."""
        np.random.seed(42)
        n = 1000
        return pd.DataFrame({
            'customerID': [f'CUST_{i}' for i in range(n)],
            'feature1': np.random.randn(n),
            'Churn': np.random.choice([0, 1], size=n, p=[0.7, 0.3])
        })
    
    def test_split_sizes(self, sample_df):
        """Splits should have correct proportions."""
        train, val, test = create_splits(
            sample_df,
            train_ratio=0.6,
            val_ratio=0.2,
            test_ratio=0.2
        )
        
        total = len(sample_df)
        assert len(train) == pytest.approx(total * 0.6, abs=5)
        assert len(val) == pytest.approx(total * 0.2, abs=5)
        assert len(test) == pytest.approx(total * 0.2, abs=5)
    
    def test_no_overlap(self, sample_df):
        """No customer should appear in multiple splits."""
        train, val, test = create_splits(sample_df)
        
        train_ids = set(train['customerID'])
        val_ids = set(val['customerID'])
        test_ids = set(test['customerID'])
        
        assert len(train_ids & val_ids) == 0
        assert len(train_ids & test_ids) == 0
        assert len(val_ids & test_ids) == 0
    
    def test_stratification(self, sample_df):
        """Churn rate should be similar across splits."""
        train, val, test = create_splits(sample_df)
        
        train_rate = train['Churn'].mean()
        val_rate = val['Churn'].mean()
        test_rate = test['Churn'].mean()
        original_rate = sample_df['Churn'].mean()
        
        # All rates should be within 5% of original
        assert abs(train_rate - original_rate) < 0.05
        assert abs(val_rate - original_rate) < 0.05
        assert abs(test_rate - original_rate) < 0.05
    
    def test_reproducibility(self, sample_df):
        """Same seed should produce same splits."""
        train1, val1, test1 = create_splits(sample_df, random_state=42)
        train2, val2, test2 = create_splits(sample_df, random_state=42)
        
        assert train1['customerID'].tolist() == train2['customerID'].tolist()
        assert val1['customerID'].tolist() == val2['customerID'].tolist()
        assert test1['customerID'].tolist() == test2['customerID'].tolist()
    
    def test_invalid_ratios(self, sample_df):
        """Should raise when ratios don't sum to 1."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            create_splits(sample_df, train_ratio=0.5, val_ratio=0.2, test_ratio=0.2)
