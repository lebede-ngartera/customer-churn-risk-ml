"""Tests for FastAPI prediction service."""

import pytest
from fastapi.testclient import TestClient
import numpy as np
import os
import sys

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Set model path environment variable before importing
os.environ["MODEL_PATH"] = os.path.join(project_root, "models", "logistic_regression.joblib")


class TestAPIEndpoints:
    """Test API endpoints."""
    
    @pytest.fixture
    def client(self):
        """Create test client with lifespan events."""
        # Import after setting environment
        from src.api.main import app
        with TestClient(app) as client:
            yield client
    
    @pytest.fixture
    def sample_customer(self):
        """Sample customer data for prediction."""
        return {
            "gender": "Female",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 24,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "Fiber optic",
            "OnlineSecurity": "No",
            "OnlineBackup": "No",
            "DeviceProtection": "No",
            "TechSupport": "No",
            "StreamingTV": "Yes",
            "StreamingMovies": "Yes",
            "Contract": "Month-to-month",
            "PaperlessBilling": "Yes",
            "PaymentMethod": "Electronic check",
            "MonthlyCharges": 89.95,
            "TotalCharges": 2159.8
        }
    
    def test_health_endpoint(self, client):
        """Health endpoint should return healthy status."""
        response = client.get("/health")
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_loaded" in data
    
    def test_predict_single_customer(self, client, sample_customer):
        """Should predict churn for a single customer."""
        response = client.post("/predict", json=sample_customer)
        
        assert response.status_code == 200
        data = response.json()
        assert "churn_prediction" in data
        assert "churn_probability" in data
        assert "risk_level" in data
        assert data["churn_prediction"] in [True, False]
        assert 0 <= data["churn_probability"] <= 1
        assert data["risk_level"] in ["low", "medium", "high"]
    
    def test_predict_batch_customers(self, client, sample_customer):
        """Should predict churn for multiple customers."""
        batch_data = {"customers": [sample_customer, sample_customer]}
        response = client.post("/predict/batch", json=batch_data)
        
        assert response.status_code == 200
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == 2
    
    def test_model_info_endpoint(self, client):
        """Model info endpoint should return model metadata."""
        response = client.get("/model/info")
        
        assert response.status_code == 200
        data = response.json()
        assert "model_type" in data
        assert "features" in data
    
    def test_invalid_customer_data(self, client):
        """Should handle invalid customer data gracefully."""
        invalid_data = {"tenure": "invalid"}  # Missing required fields
        response = client.post("/predict", json=invalid_data)
        
        # Should return validation error
        assert response.status_code == 422
    
    def test_prediction_probability_consistency(self, client, sample_customer):
        """Prediction and probability should be consistent."""
        response = client.post("/predict", json=sample_customer)
        data = response.json()
        
        # If probability > 0.5, prediction should be True (churn)
        if data["churn_probability"] > 0.5:
            assert data["churn_prediction"] is True
        else:
            assert data["churn_prediction"] is False
    
    def test_risk_level_thresholds(self, client, sample_customer):
        """Risk levels should match probability thresholds."""
        response = client.post("/predict", json=sample_customer)
        data = response.json()
        
        prob = data["churn_probability"]
        risk = data["risk_level"]
        
        # Thresholds: low < 0.4, medium 0.4-0.7, high >= 0.7
        if prob < 0.4:
            assert risk == "low"
        elif prob < 0.7:
            assert risk == "medium"
        else:
            assert risk == "high"


class TestCustomerFeatureValidation:
    """Test input validation for customer features."""
    
    @pytest.fixture
    def client(self):
        """Create test client with lifespan events."""
        from src.api.main import app
        with TestClient(app) as client:
            yield client
    
    def test_accepts_valid_gender(self, client):
        """Should accept valid gender values."""
        from src.api.main import CustomerFeatures
        
        # Valid genders
        valid_data = {
            "gender": "Male",
            "SeniorCitizen": 0,
            "Partner": "Yes",
            "Dependents": "No",
            "tenure": 12,
            "PhoneService": "Yes",
            "MultipleLines": "No",
            "InternetService": "DSL",
            "OnlineSecurity": "Yes",
            "OnlineBackup": "Yes",
            "DeviceProtection": "Yes",
            "TechSupport": "Yes",
            "StreamingTV": "No",
            "StreamingMovies": "No",
            "Contract": "One year",
            "PaperlessBilling": "No",
            "PaymentMethod": "Credit card (automatic)",
            "MonthlyCharges": 50.0,
            "TotalCharges": 600.0
        }
        
        response = client.post("/predict", json=valid_data)
        assert response.status_code == 200
    
    def test_numeric_bounds(self, client):
        """Should validate numeric feature bounds."""
        from src.api.main import CustomerFeatures
        
        # tenure should be non-negative
        customer = CustomerFeatures(
            gender="Female",
            SeniorCitizen=0,
            Partner="No",
            Dependents="No",
            tenure=0,  # Edge case: new customer
            PhoneService="Yes",
            MultipleLines="No",
            InternetService="No",
            OnlineSecurity="No internet service",
            OnlineBackup="No internet service",
            DeviceProtection="No internet service",
            TechSupport="No internet service",
            StreamingTV="No internet service",
            StreamingMovies="No internet service",
            Contract="Month-to-month",
            PaperlessBilling="Yes",
            PaymentMethod="Electronic check",
            MonthlyCharges=20.0,
            TotalCharges=0.0
        )
        
        # Should be valid
        assert customer.tenure >= 0
        assert customer.MonthlyCharges >= 0
        assert customer.TotalCharges >= 0
