"""Tests for model explainability module."""

import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

from src.models.explainability import (
    get_logistic_regression_coefficients,
    get_tree_feature_importance,
    get_feature_importance,
    explain_prediction,
    generate_plain_english_explanation
)


@pytest.fixture
def sample_data():
    """Create sample data for testing."""
    np.random.seed(42)
    n_samples = 100
    return pd.DataFrame({
        'feature_a': np.random.randn(n_samples),
        'feature_b': np.random.randn(n_samples),
        'feature_c': np.random.randn(n_samples)
    })


@pytest.fixture
def sample_target(sample_data):
    """Create sample target based on data."""
    np.random.seed(42)
    # Create target correlated with features
    probs = 1 / (1 + np.exp(-(sample_data['feature_a'] + sample_data['feature_b'])))
    return (np.random.rand(len(sample_data)) < probs).astype(int)


@pytest.fixture
def feature_names():
    """Feature names after preprocessing."""
    return ['feature_a', 'feature_b', 'feature_c']


@pytest.fixture
def simple_preprocessor(sample_data):
    """Create simple preprocessor."""
    return ColumnTransformer(
        transformers=[
            ('scaler', StandardScaler(), ['feature_a', 'feature_b', 'feature_c'])
        ],
        remainder='drop'
    )


@pytest.fixture
def trained_logistic_model(sample_data, sample_target, simple_preprocessor):
    """Create trained logistic regression model."""
    # Preprocess data first
    simple_preprocessor.fit(sample_data)
    X_transformed = simple_preprocessor.transform(sample_data)
    
    # Train model on preprocessed data
    model = LogisticRegression(random_state=42)
    model.fit(X_transformed, sample_target)
    return model


@pytest.fixture
def trained_rf_model(sample_data, sample_target, simple_preprocessor):
    """Create trained random forest model."""
    simple_preprocessor.fit(sample_data)
    X_transformed = simple_preprocessor.transform(sample_data)
    
    model = RandomForestClassifier(n_estimators=10, random_state=42)
    model.fit(X_transformed, sample_target)
    return model


@pytest.fixture
def trained_gb_model(sample_data, sample_target, simple_preprocessor):
    """Create trained gradient boosting model."""
    simple_preprocessor.fit(sample_data)
    X_transformed = simple_preprocessor.transform(sample_data)
    
    model = GradientBoostingClassifier(n_estimators=10, random_state=42)
    model.fit(X_transformed, sample_target)
    return model


@pytest.fixture
def trained_pipeline(sample_data, sample_target, simple_preprocessor):
    """Create trained pipeline with logistic regression."""
    pipeline = Pipeline([
        ('preprocessor', simple_preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])
    pipeline.fit(sample_data, sample_target)
    return pipeline


@pytest.fixture
def trained_preprocessor_and_model(sample_data, sample_target, simple_preprocessor):
    """Create fitted preprocessor and model separately."""
    # Fit preprocessor
    preprocessor = simple_preprocessor
    preprocessor.fit(sample_data)
    X_transformed = preprocessor.transform(sample_data)
    
    # Fit model
    model = LogisticRegression(random_state=42)
    model.fit(X_transformed, sample_target)
    
    return preprocessor, model


class TestLogisticRegressionCoefficients:
    """Test logistic regression coefficient extraction."""
    
    def test_extracts_coefficients(self, trained_logistic_model, feature_names):
        """Should extract coefficients from logistic regression."""
        coef_df = get_logistic_regression_coefficients(
            trained_logistic_model, feature_names
        )
        
        assert isinstance(coef_df, pd.DataFrame)
        assert 'feature' in coef_df.columns
        assert 'coefficient' in coef_df.columns
        assert 'abs_coefficient' in coef_df.columns
        assert len(coef_df) == 3  # 3 features
    
    def test_sorted_by_importance(self, trained_logistic_model, feature_names):
        """Should sort by absolute coefficient descending."""
        coef_df = get_logistic_regression_coefficients(
            trained_logistic_model, feature_names
        )
        
        assert coef_df['abs_coefficient'].is_monotonic_decreasing
    
    def test_has_direction_column(self, trained_logistic_model, feature_names):
        """Should include direction interpretation."""
        coef_df = get_logistic_regression_coefficients(
            trained_logistic_model, feature_names
        )
        
        assert 'direction' in coef_df.columns
        assert all(d in ['Increases Churn', 'Decreases Churn'] 
                   for d in coef_df['direction'])


class TestTreeFeatureImportance:
    """Test tree-based feature importance extraction."""
    
    def test_extracts_rf_importance(self, trained_rf_model, feature_names):
        """Should extract importance from random forest."""
        imp_df = get_tree_feature_importance(trained_rf_model, feature_names)
        
        assert isinstance(imp_df, pd.DataFrame)
        assert 'feature' in imp_df.columns
        assert 'importance' in imp_df.columns
        assert len(imp_df) == 3
    
    def test_extracts_gb_importance(self, trained_gb_model, feature_names):
        """Should extract importance from gradient boosting."""
        imp_df = get_tree_feature_importance(trained_gb_model, feature_names)
        
        assert isinstance(imp_df, pd.DataFrame)
        assert len(imp_df) == 3
    
    def test_sorted_by_importance(self, trained_rf_model, feature_names):
        """Should sort by importance descending."""
        imp_df = get_tree_feature_importance(trained_rf_model, feature_names)
        
        assert imp_df['importance'].is_monotonic_decreasing
    
    def test_raises_for_wrong_model(self, trained_logistic_model, feature_names):
        """Should raise error for non-tree model."""
        with pytest.raises(ValueError, match="feature_importances_"):
            get_tree_feature_importance(trained_logistic_model, feature_names)


class TestGetFeatureImportance:
    """Test generic feature importance extraction."""
    
    def test_handles_logistic_regression(self, trained_logistic_model, feature_names):
        """Should handle logistic regression automatically."""
        imp_df = get_feature_importance(trained_logistic_model, feature_names)
        
        assert isinstance(imp_df, pd.DataFrame)
        assert 'feature' in imp_df.columns
        assert 'importance' in imp_df.columns
    
    def test_handles_random_forest(self, trained_rf_model, feature_names):
        """Should handle random forest automatically."""
        imp_df = get_feature_importance(trained_rf_model, feature_names)
        
        assert isinstance(imp_df, pd.DataFrame)
    
    def test_handles_gradient_boosting(self, trained_gb_model, feature_names):
        """Should handle gradient boosting automatically."""
        imp_df = get_feature_importance(trained_gb_model, feature_names)
        
        assert isinstance(imp_df, pd.DataFrame)


class TestExplainPrediction:
    """Test individual prediction explanation."""
    
    def test_explains_single_prediction(self, trained_preprocessor_and_model, sample_data, feature_names):
        """Should explain a single prediction."""
        preprocessor, model = trained_preprocessor_and_model
        single_sample = sample_data.iloc[[0]]
        explanation = explain_prediction(model, preprocessor, single_sample, feature_names)
        
        assert 'churn_probability' in explanation
        assert 'prediction' in explanation
        assert 'top_features' in explanation
        assert 'risk_level' in explanation
    
    def test_prediction_values(self, trained_preprocessor_and_model, sample_data, feature_names):
        """Should have valid prediction values."""
        preprocessor, model = trained_preprocessor_and_model
        single_sample = sample_data.iloc[[0]]
        explanation = explain_prediction(model, preprocessor, single_sample, feature_names)
        
        assert explanation['prediction'] in ['Will Churn', 'Will Stay']
        assert 0 <= explanation['churn_probability'] <= 1
    
    def test_top_features_exist(self, trained_preprocessor_and_model, sample_data, feature_names):
        """Top features should be populated."""
        preprocessor, model = trained_preprocessor_and_model
        single_sample = sample_data.iloc[[0]]
        explanation = explain_prediction(model, preprocessor, single_sample, feature_names)
        
        # Check that top_features is a list
        assert isinstance(explanation['top_features'], list)


class TestGeneratePlainEnglishExplanation:
    """Test plain English explanation generation."""
    
    def test_generates_text(self, trained_preprocessor_and_model, sample_data, feature_names):
        """Should generate readable text explanation."""
        preprocessor, model = trained_preprocessor_and_model
        single_sample = sample_data.iloc[[0]]
        explanation = explain_prediction(model, preprocessor, single_sample, feature_names)
        text = generate_plain_english_explanation(explanation)
        
        assert isinstance(text, str)
        assert len(text) > 0
    
    def test_mentions_probability(self, trained_preprocessor_and_model, sample_data, feature_names):
        """Should mention the probability."""
        preprocessor, model = trained_preprocessor_and_model
        single_sample = sample_data.iloc[[0]]
        explanation = explain_prediction(model, preprocessor, single_sample, feature_names)
        text = generate_plain_english_explanation(explanation)
        
        # Should mention percentage
        assert '%' in text or 'probability' in text.lower()
    
    def test_mentions_churn_assessment(self, trained_preprocessor_and_model, sample_data, feature_names):
        """Should mention churn risk assessment."""
        preprocessor, model = trained_preprocessor_and_model
        single_sample = sample_data.iloc[[0]]
        explanation = explain_prediction(model, preprocessor, single_sample, feature_names)
        text = generate_plain_english_explanation(explanation)
        
        # Should assess churn risk
        assert ('churn' in text.lower() or 
                'stay' in text.lower() or 
                'risk' in text.lower() or
                'likely' in text.lower())
