"""
Tests for model training and evaluation.
"""

import pytest
import numpy as np
from pathlib import Path
import sys
import tempfile

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.training import (
    create_logistic_regression,
    create_random_forest,
    create_gradient_boosting,
    train_model,
    save_model,
    load_model
)
from src.models.evaluation import (
    calculate_metrics,
    calculate_confusion_matrix,
    evaluate_model,
    compare_models,
    find_optimal_threshold
)


class TestModelCreation:
    """Tests for model creation functions."""
    
    def test_create_logistic_regression(self):
        """Logistic regression should be created with expected params."""
        model = create_logistic_regression(C=0.5)
        
        assert model.C == 0.5
        assert model.max_iter == 1000
        assert hasattr(model, 'fit')
    
    def test_create_random_forest(self):
        """Random forest should be created with expected params."""
        model = create_random_forest(n_estimators=50, max_depth=5)
        
        assert model.n_estimators == 50
        assert model.max_depth == 5
        assert hasattr(model, 'fit')
    
    def test_create_gradient_boosting(self):
        """Gradient boosting should be created with expected params."""
        model = create_gradient_boosting(n_estimators=50, learning_rate=0.05)
        
        assert model.n_estimators == 50
        assert model.learning_rate == 0.05
        assert hasattr(model, 'fit')


class TestModelTraining:
    """Tests for model training."""
    
    @pytest.fixture
    def sample_data(self):
        """Create sample training data."""
        np.random.seed(42)
        X = np.random.randn(100, 10)
        y = (X[:, 0] + X[:, 1] > 0).astype(int)  # Simple rule
        return X, y
    
    def test_train_logistic_regression(self, sample_data):
        """Logistic regression should train successfully."""
        X, y = sample_data
        model = create_logistic_regression()
        
        trained = train_model(model, X, y)
        
        assert hasattr(trained, 'coef_')
        assert trained.coef_.shape[1] == 10
    
    def test_train_random_forest(self, sample_data):
        """Random forest should train successfully."""
        X, y = sample_data
        model = create_random_forest(n_estimators=10)
        
        trained = train_model(model, X, y)
        
        assert hasattr(trained, 'feature_importances_')
        assert len(trained.feature_importances_) == 10
    
    def test_trained_model_can_predict(self, sample_data):
        """Trained model should make predictions."""
        X, y = sample_data
        model = create_logistic_regression()
        trained = train_model(model, X, y)
        
        predictions = trained.predict(X)
        probabilities = trained.predict_proba(X)
        
        assert len(predictions) == 100
        assert probabilities.shape == (100, 2)
        assert np.all((predictions == 0) | (predictions == 1))


class TestModelSaveLoad:
    """Tests for model persistence."""
    
    @pytest.fixture
    def trained_model(self):
        """Create a trained model."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = create_logistic_regression()
        return train_model(model, X, y)
    
    def test_save_and_load_model(self, trained_model):
        """Model should be saveable and loadable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test_model.joblib"
            
            # Save
            save_model(trained_model, str(filepath))
            assert filepath.exists()
            
            # Load
            loaded_model, preprocessor = load_model(str(filepath))
            
            # Should have same coefficients
            np.testing.assert_array_almost_equal(
                trained_model.coef_,
                loaded_model.coef_
            )


class TestEvaluation:
    """Tests for model evaluation."""
    
    def test_calculate_metrics(self):
        """Metrics should be calculated correctly."""
        y_true = np.array([0, 0, 1, 1, 1])
        y_pred = np.array([0, 1, 1, 1, 0])
        y_prob = np.array([0.2, 0.6, 0.8, 0.9, 0.3])
        
        metrics = calculate_metrics(y_true, y_pred, y_prob)
        
        assert 'roc_auc' in metrics
        assert 'precision' in metrics
        assert 'recall' in metrics
        assert 'f1' in metrics
        
        # Check ranges
        for name, value in metrics.items():
            assert 0 <= value <= 1, f"{name} should be between 0 and 1"
    
    def test_confusion_matrix(self):
        """Confusion matrix should have correct components."""
        y_true = np.array([0, 0, 0, 1, 1])
        y_pred = np.array([0, 0, 1, 1, 0])
        
        cm = calculate_confusion_matrix(y_true, y_pred)
        
        assert cm['true_negatives'] == 2
        assert cm['false_positives'] == 1
        assert cm['false_negatives'] == 1
        assert cm['true_positives'] == 1
    
    def test_evaluate_model(self):
        """Full evaluation should return expected structure."""
        np.random.seed(42)
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)
        
        model = create_logistic_regression()
        model.fit(X, y)
        
        results = evaluate_model(model, X, y)
        
        assert 'metrics' in results
        assert 'confusion_matrix' in results
        assert 'threshold' in results
        assert results['threshold'] == 0.5
    
    def test_compare_models(self):
        """Model comparison should return sorted DataFrame."""
        results = {
            'Model A': {'metrics': {'roc_auc': 0.75, 'precision': 0.7, 'recall': 0.6, 'f1': 0.65}},
            'Model B': {'metrics': {'roc_auc': 0.85, 'precision': 0.8, 'recall': 0.7, 'f1': 0.75}}
        }
        
        df = compare_models(results)
        
        # Should be sorted by ROC-AUC descending
        assert df.index[0] == 'Model B'
        assert df.loc['Model B', 'roc_auc'] == 0.85


class TestThresholdOptimization:
    """Tests for threshold optimization."""
    
    def test_find_optimal_threshold(self):
        """Optimal threshold should maximize the specified metric."""
        np.random.seed(42)
        X = np.random.randn(200, 5)
        y = (X[:, 0] > 0).astype(int)
        
        model = create_logistic_regression()
        model.fit(X, y)
        
        threshold, score = find_optimal_threshold(model, X, y, metric='f1')
        
        assert 0.1 <= threshold <= 0.9
        assert 0 <= score <= 1
