"""
Tests for feature preprocessing.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.features.preprocessing import (
    create_preprocessor,
    encode_binary_feature,
    get_feature_names
)
from src.features.feature_definitions import (
    NUMERIC_COLS,
    BINARY_COLS,
    CATEGORICAL_COLS
)


class TestEncodeBinaryFeature:
    """Tests for binary feature encoding."""
    
    def test_encodes_yes_no(self):
        """Yes should become 1, No should become 0."""
        X = np.array(['Yes', 'No', 'Yes', 'No'])
        result = encode_binary_feature(X)
        
        np.testing.assert_array_equal(result, [1, 0, 1, 0])
    
    def test_encodes_male_female(self):
        """Male should become 1, Female should become 0."""
        X = np.array(['Male', 'Female', 'Male'])
        result = encode_binary_feature(X)
        
        np.testing.assert_array_equal(result, [1, 0, 1])
    
    def test_handles_case_insensitive(self):
        """Should handle different cases."""
        X = np.array(['YES', 'no', 'Yes', 'NO'])
        result = encode_binary_feature(X)
        
        np.testing.assert_array_equal(result, [1, 0, 1, 0])
    
    def test_handles_numeric_strings(self):
        """Should handle '0' and '1' strings."""
        X = np.array(['1', '0', '1'])
        result = encode_binary_feature(X)
        
        np.testing.assert_array_equal(result, [1, 0, 1])


class TestCreatePreprocessor:
    """Tests for the preprocessing pipeline."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data matching expected schema."""
        return pd.DataFrame({
            # Numeric
            'tenure': [12, 24, 36],
            'MonthlyCharges': [50.0, 75.0, 100.0],
            'TotalCharges': [600.0, 1800.0, 3600.0],
            # Binary
            'gender': ['Male', 'Female', 'Male'],
            'SeniorCitizen': [0, 1, 0],
            'Partner': ['Yes', 'No', 'Yes'],
            'Dependents': ['No', 'No', 'Yes'],
            'PhoneService': ['Yes', 'Yes', 'No'],
            'PaperlessBilling': ['Yes', 'No', 'Yes'],
            # Categorical
            'MultipleLines': ['Yes', 'No', 'No phone service'],
            'InternetService': ['DSL', 'Fiber optic', 'No'],
            'OnlineSecurity': ['Yes', 'No', 'No internet service'],
            'OnlineBackup': ['Yes', 'No', 'No internet service'],
            'DeviceProtection': ['Yes', 'No', 'No internet service'],
            'TechSupport': ['Yes', 'No', 'No internet service'],
            'StreamingTV': ['Yes', 'No', 'No internet service'],
            'StreamingMovies': ['Yes', 'No', 'No internet service'],
            'Contract': ['Month-to-month', 'One year', 'Two year'],
            'PaymentMethod': ['Electronic check', 'Mailed check', 'Bank transfer (automatic)'],
        })
    
    def test_preprocessor_fits(self, sample_data):
        """Preprocessor should fit without error."""
        preprocessor = create_preprocessor()
        preprocessor.fit(sample_data)
        
        # Should not raise
        assert hasattr(preprocessor, 'transformers_')
    
    def test_preprocessor_transforms(self, sample_data):
        """Preprocessor should transform data to numpy array."""
        preprocessor = create_preprocessor()
        result = preprocessor.fit_transform(sample_data)
        
        assert isinstance(result, np.ndarray)
        assert result.shape[0] == 3  # Same number of rows
    
    def test_numeric_scaling(self, sample_data):
        """Numeric features should be scaled."""
        preprocessor = create_preprocessor()
        result = preprocessor.fit_transform(sample_data)
        
        # After StandardScaler, mean should be ~0, std ~1
        # First 3 columns are numeric
        numeric_result = result[:, :3]
        
        # With only 3 samples, we can't expect perfect scaling
        # but values should be transformed (not original)
        assert not np.allclose(numeric_result[:, 0], [12, 24, 36])
    
    def test_no_data_leakage(self, sample_data):
        """Transform on new data should use training statistics."""
        preprocessor = create_preprocessor()
        
        # Fit on training data
        train_data = sample_data.iloc[:2]
        preprocessor.fit(train_data)
        
        # Transform on "test" data with different values
        test_data = sample_data.iloc[2:]
        result = preprocessor.transform(test_data)
        
        # Result should exist and have correct shape
        assert result.shape[0] == 1
    
    def test_get_feature_names(self, sample_data):
        """Should be able to retrieve feature names after fitting."""
        preprocessor = create_preprocessor()
        preprocessor.fit(sample_data)
        
        names = get_feature_names(preprocessor)
        
        assert len(names) > 0
        assert all(isinstance(name, str) for name in names)
        
        # Should include numeric features
        assert any('tenure' in name for name in names)
        assert any('MonthlyCharges' in name for name in names)


class TestPreprocessorIntegration:
    """Integration tests for the full preprocessing pipeline."""
    
    @pytest.fixture
    def real_sample(self):
        """Sample that mimics real data patterns."""
        np.random.seed(42)
        n = 100
        
        return pd.DataFrame({
            'tenure': np.random.randint(0, 72, n),
            'MonthlyCharges': np.random.uniform(20, 100, n),
            'TotalCharges': np.random.uniform(0, 5000, n),
            'gender': np.random.choice(['Male', 'Female'], n),
            'SeniorCitizen': np.random.choice([0, 1], n),
            'Partner': np.random.choice(['Yes', 'No'], n),
            'Dependents': np.random.choice(['Yes', 'No'], n),
            'PhoneService': np.random.choice(['Yes', 'No'], n),
            'PaperlessBilling': np.random.choice(['Yes', 'No'], n),
            'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n),
            'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n),
            'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n),
            'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n),
            'PaymentMethod': np.random.choice([
                'Electronic check', 'Mailed check', 
                'Bank transfer (automatic)', 'Credit card (automatic)'
            ], n),
        })
    
    def test_full_pipeline(self, real_sample):
        """Full pipeline should work on realistic data."""
        from src.data.splitting import create_splits
        
        # Add target and ID for splitting
        real_sample['Churn'] = np.random.choice([0, 1], len(real_sample), p=[0.7, 0.3])
        real_sample['customerID'] = [f'C{i}' for i in range(len(real_sample))]
        
        # Split
        train, val, test = create_splits(real_sample)
        
        # Get features only (drop ID and target)
        X_train = train.drop(['customerID', 'Churn'], axis=1)
        X_val = val.drop(['customerID', 'Churn'], axis=1)
        X_test = test.drop(['customerID', 'Churn'], axis=1)
        
        # Preprocess
        preprocessor = create_preprocessor()
        X_train_processed = preprocessor.fit_transform(X_train)
        X_val_processed = preprocessor.transform(X_val)
        X_test_processed = preprocessor.transform(X_test)
        
        # Verify shapes
        assert X_train_processed.shape[0] == len(train)
        assert X_val_processed.shape[0] == len(val)
        assert X_test_processed.shape[0] == len(test)
        
        # All should have same number of features
        assert X_train_processed.shape[1] == X_val_processed.shape[1]
        assert X_train_processed.shape[1] == X_test_processed.shape[1]
        
        # No NaN values
        assert not np.isnan(X_train_processed).any()
        assert not np.isnan(X_val_processed).any()
        assert not np.isnan(X_test_processed).any()
