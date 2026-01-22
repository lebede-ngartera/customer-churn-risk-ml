"""
FastAPI service for customer churn prediction.

This module provides a REST API for making churn predictions.
It loads a trained model and preprocessor, then exposes endpoints
for single and batch predictions.

Usage:
    uvicorn src.api.main:app --reload
    
Endpoints:
    GET  /health          - Health check
    POST /predict         - Single customer prediction
    POST /predict/batch   - Batch predictions
    GET  /model/info      - Model information
"""

import os
import sys
from pathlib import Path
from typing import List, Optional
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, ConfigDict
import pandas as pd
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.features import FEATURE_COLS, NUMERIC_COLS, BINARY_COLS, CATEGORICAL_COLS

# Global model storage
model_artifacts = {}


def get_model_path():
    """Get the model path, checking multiple locations."""
    # Check environment variable first
    if os.environ.get("MODEL_PATH"):
        return Path(os.environ["MODEL_PATH"])
    
    # Check relative to project root
    model_path = project_root / "models" / "logistic_regression.joblib"
    if model_path.exists():
        return model_path
    
    # Check relative to CWD
    cwd_path = Path.cwd() / "models" / "logistic_regression.joblib"
    if cwd_path.exists():
        return cwd_path
        
    return None


def load_model_artifacts():
    """Load model and preprocessor from disk."""
    from src.models import load_model
    
    model_path = get_model_path()
    
    if model_path is None or not model_path.exists():
        return None  # Model not available
    
    model, preprocessor = load_model(str(model_path))
    
    return {
        'model': model,
        'preprocessor': preprocessor,
        'model_type': type(model).__name__,
        'model_path': str(model_path)
    }


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup."""
    global model_artifacts
    print("Loading model artifacts...")
    artifacts = load_model_artifacts()
    if artifacts:
        model_artifacts.update(artifacts)
        print(f"Model loaded: {model_artifacts['model_type']}")
    else:
        print("Warning: Model not found. Prediction endpoints will return 503.")
    yield
    # Cleanup on shutdown
    model_artifacts.clear()


# Initialize FastAPI app
app = FastAPI(
    title="Customer Churn Prediction API",
    description="Predict customer churn probability using machine learning",
    version="1.0.0",
    lifespan=lifespan
)


# ============================================================================
# Pydantic Models (Request/Response Schemas)
# ============================================================================

class CustomerFeatures(BaseModel):
    """Input features for a single customer."""
    
    # Numeric features
    tenure: int = Field(..., ge=0, le=100, description="Months with company")
    MonthlyCharges: float = Field(..., ge=0, description="Monthly charge amount")
    TotalCharges: float = Field(..., ge=0, description="Total charges to date")
    
    # Binary features
    gender: str = Field(..., pattern="^(Male|Female)$")
    SeniorCitizen: int = Field(..., ge=0, le=1)
    Partner: str = Field(..., pattern="^(Yes|No)$")
    Dependents: str = Field(..., pattern="^(Yes|No)$")
    PhoneService: str = Field(..., pattern="^(Yes|No)$")
    PaperlessBilling: str = Field(..., pattern="^(Yes|No)$")
    
    # Categorical features
    MultipleLines: str = Field(..., description="Yes/No/No phone service")
    InternetService: str = Field(..., description="DSL/Fiber optic/No")
    OnlineSecurity: str = Field(..., description="Yes/No/No internet service")
    OnlineBackup: str = Field(..., description="Yes/No/No internet service")
    DeviceProtection: str = Field(..., description="Yes/No/No internet service")
    TechSupport: str = Field(..., description="Yes/No/No internet service")
    StreamingTV: str = Field(..., description="Yes/No/No internet service")
    StreamingMovies: str = Field(..., description="Yes/No/No internet service")
    Contract: str = Field(..., description="Month-to-month/One year/Two year")
    PaymentMethod: str = Field(..., description="Payment method")
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "tenure": 24,
                "MonthlyCharges": 65.5,
                "TotalCharges": 1572.0,
                "gender": "Male",
                "SeniorCitizen": 0,
                "Partner": "Yes",
                "Dependents": "No",
                "PhoneService": "Yes",
                "PaperlessBilling": "Yes",
                "MultipleLines": "No",
                "InternetService": "Fiber optic",
                "OnlineSecurity": "No",
                "OnlineBackup": "Yes",
                "DeviceProtection": "No",
                "TechSupport": "No",
                "StreamingTV": "Yes",
                "StreamingMovies": "Yes",
                "Contract": "Month-to-month",
                "PaymentMethod": "Electronic check"
            }
        }
    )


class PredictionResponse(BaseModel):
    """Response for a single prediction."""
    
    churn_probability: float = Field(..., description="Probability of churn (0-1)")
    churn_prediction: bool = Field(..., description="Binary prediction")
    risk_level: str = Field(..., description="Risk category")
    confidence: float = Field(..., description="Prediction confidence")


class BatchPredictionRequest(BaseModel):
    """Request for batch predictions."""
    
    customers: List[CustomerFeatures]


class BatchPredictionResponse(BaseModel):
    """Response for batch predictions."""
    
    predictions: List[PredictionResponse]
    total_customers: int
    high_risk_count: int
    medium_risk_count: int
    low_risk_count: int


class ModelInfo(BaseModel):
    """Model metadata."""
    
    model_type: str
    model_path: str
    feature_count: int
    features: List[str]


class HealthResponse(BaseModel):
    """Health check response."""
    
    status: str
    model_loaded: bool


# ============================================================================
# Helper Functions
# ============================================================================

def get_risk_level(probability: float) -> str:
    """Categorize probability into risk level."""
    if probability >= 0.7:
        return "high"
    elif probability >= 0.4:
        return "medium"
    else:
        return "low"


def features_to_dataframe(features: CustomerFeatures) -> pd.DataFrame:
    """Convert Pydantic model to DataFrame."""
    return pd.DataFrame([features.model_dump()])


def make_prediction(features_df: pd.DataFrame) -> dict:
    """Make a prediction for a single customer."""
    model = model_artifacts['model']
    preprocessor = model_artifacts['preprocessor']
    
    # Ensure columns are in correct order
    features_df = features_df[FEATURE_COLS]
    
    # Preprocess
    X = preprocessor.transform(features_df)
    
    # Predict
    prob = model.predict_proba(X)[0, 1]
    pred = prob >= 0.5
    
    return {
        'churn_probability': round(float(prob), 4),
        'churn_prediction': bool(pred),
        'risk_level': get_risk_level(prob),
        'confidence': round(float(max(prob, 1 - prob)), 4)
    }


# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Check API health and model status."""
    return {
        "status": "healthy",
        "model_loaded": 'model' in model_artifacts
    }


@app.get("/model/info", response_model=ModelInfo, tags=["Model"])
async def get_model_info():
    """Get information about the loaded model."""
    if 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    return {
        "model_type": model_artifacts['model_type'],
        "model_path": model_artifacts['model_path'],
        "feature_count": len(FEATURE_COLS),
        "features": FEATURE_COLS
    }


@app.post("/predict", response_model=PredictionResponse, tags=["Predictions"])
async def predict_single(customer: CustomerFeatures):
    """
    Predict churn probability for a single customer.
    
    Returns the probability of churn, binary prediction, risk level,
    and confidence score.
    """
    if 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        features_df = features_to_dataframe(customer)
        result = make_prediction(features_df)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["Predictions"])
async def predict_batch(request: BatchPredictionRequest):
    """
    Predict churn probability for multiple customers.
    
    Returns individual predictions plus summary statistics.
    """
    if 'model' not in model_artifacts:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    if len(request.customers) == 0:
        raise HTTPException(status_code=400, detail="No customers provided")
    
    if len(request.customers) > 1000:
        raise HTTPException(status_code=400, detail="Maximum 1000 customers per batch")
    
    try:
        predictions = []
        risk_counts = {"high": 0, "medium": 0, "low": 0}
        
        for customer in request.customers:
            features_df = features_to_dataframe(customer)
            result = make_prediction(features_df)
            predictions.append(result)
            risk_counts[result['risk_level']] += 1
        
        return {
            "predictions": predictions,
            "total_customers": len(predictions),
            "high_risk_count": risk_counts["high"],
            "medium_risk_count": risk_counts["medium"],
            "low_risk_count": risk_counts["low"]
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


# ============================================================================
# Run directly for development
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
