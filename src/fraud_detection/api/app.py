"""
FastAPI application for real-time fraud detection inference.
Provides REST API endpoints for fraud detection predictions.
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional
import pandas as pd
import numpy as np
import logging
from datetime import datetime
import asyncio
import os
from pathlib import Path
import json

# Prometheus monitoring
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from fastapi import Response
from prometheus_fastapi_instrumentator import Instrumentator

# Local imports
from ..models.xgboost_model import XGBoostFraudDetector
from ..models.lstm_model import LSTMFraudDetectorWrapper
from ..models.tab_transformer import TabTransformerWrapper
from ..utils.config import load_config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Prometheus metrics
PREDICTION_COUNTER = Counter('fraud_predictions_total', 'Total fraud predictions', ['model', 'prediction'])
PREDICTION_LATENCY = Histogram('fraud_prediction_latency_seconds', 'Prediction latency', ['model'])
ACTIVE_MODELS = Gauge('active_models', 'Number of active models')

# Pydantic models for API
class TransactionFeatures(BaseModel):
    """Individual transaction features."""
    customer_id: str = Field(..., description="Customer identifier")
    amount: float = Field(..., description="Transaction amount")
    merchant_category: str = Field(..., description="Merchant category")
    location: str = Field(..., description="Transaction location")
    day_of_week: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    hour_of_day: int = Field(..., ge=0, le=23, description="Hour of day")
    is_weekend: Optional[int] = Field(0, description="Is weekend transaction")
    is_night: Optional[int] = Field(0, description="Is night transaction")
    
    # Additional features (optional)
    avg_amount: Optional[float] = Field(None, description="Customer average amount")
    std_amount: Optional[float] = Field(None, description="Customer amount std dev")
    transaction_count: Optional[int] = Field(None, description="Customer transaction count")
    unique_categories: Optional[int] = Field(None, description="Unique merchant categories")
    amount_zscore: Optional[float] = Field(None, description="Amount z-score")
    time_since_last: Optional[float] = Field(None, description="Hours since last transaction")
    transactions_last_hour: Optional[int] = Field(0, description="Transactions in last hour")


class PredictionRequest(BaseModel):
    """Request model for fraud prediction."""
    transactions: List[TransactionFeatures]
    model_type: Optional[str] = Field("xgboost", description="Model type: xgboost, lstm, or tab_transformer")
    threshold: Optional[float] = Field(0.5, description="Classification threshold")


class PredictionResponse(BaseModel):
    """Response model for fraud prediction."""
    predictions: List[Dict[str, Any]]
    model_used: str
    processing_time: float
    timestamp: str


class BatchPredictionRequest(BaseModel):
    """Request model for batch prediction."""
    data: Dict[str, Any]  # JSON representation of transaction data
    model_type: Optional[str] = Field("xgboost", description="Model type")
    
    
class HealthResponse(BaseModel):
    """Health check response."""
    status: str
    models_loaded: Dict[str, bool]
    timestamp: str


# FastAPI app
app = FastAPI(
    title="Real-time Fraud Detection API",
    description="ML-powered fraud detection with XGBoost, LSTM, and TabTransformer models",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add Prometheus instrumentation
instrumentator = Instrumentator()
instrumentator.instrument(app).expose(app)

# Model storage
models = {
    "xgboost": None,
    "lstm": None,
    "tab_transformer": None
}

config = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and configuration on startup."""
    global config, models
    
    logger.info("Starting Fraud Detection API...")
    
    # Load configuration
    config = load_config()
    
    # Load models
    await load_models()
    
    logger.info("Fraud Detection API started successfully!")


async def load_models():
    """Load trained models."""
    global models
    
    model_path = Path("models")
    if not model_path.exists():
        logger.warning("Models directory not found. Models will not be available.")
        return
    
    # Load XGBoost model
    xgb_path = model_path / "xgboost_model.joblib"
    if xgb_path.exists():
        try:
            models["xgboost"] = XGBoostFraudDetector(config)
            models["xgboost"].load_model(str(xgb_path))
            logger.info("XGBoost model loaded successfully")
            ACTIVE_MODELS.inc()
        except Exception as e:
            logger.error(f"Failed to load XGBoost model: {e}")
    
    # Load LSTM model
    lstm_path = model_path / "lstm_model.pt"
    if lstm_path.exists():
        try:
            models["lstm"] = LSTMFraudDetectorWrapper(config)
            models["lstm"].load_model(str(lstm_path))
            logger.info("LSTM model loaded successfully")
            ACTIVE_MODELS.inc()
        except Exception as e:
            logger.error(f"Failed to load LSTM model: {e}")
    
    # Load TabTransformer model
    tab_path = model_path / "tab_transformer_model.pt"
    if tab_path.exists():
        try:
            models["tab_transformer"] = TabTransformerWrapper(config)
            models["tab_transformer"].load_model(str(tab_path))
            logger.info("TabTransformer model loaded successfully")
            ACTIVE_MODELS.inc()
        except Exception as e:
            logger.error(f"Failed to load TabTransformer model: {e}")


def prepare_dataframe(transactions: List[TransactionFeatures]) -> pd.DataFrame:
    """Convert transaction features to DataFrame."""
    data = []
    for trans in transactions:
        trans_dict = trans.dict()
        # Add transaction_time for compatibility
        trans_dict['transaction_time'] = datetime.now()
        data.append(trans_dict)
    
    df = pd.DataFrame(data)
    
    # Fill missing optional features with defaults
    optional_features = {
        'avg_amount': df['amount'].mean(),
        'std_amount': df['amount'].std(),
        'transaction_count': 1,
        'unique_categories': 1,
        'amount_zscore': 0,
        'time_since_last': 24,
        'transactions_last_hour': 0
    }
    
    for feature, default_value in optional_features.items():
        if feature not in df.columns:
            df[feature] = default_value
        else:
            df[feature] = df[feature].fillna(default_value)
    
    return df


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint."""
    return {"message": "Fraud Detection API", "version": "1.0.0"}


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_status = {
        model_name: model is not None and getattr(model, 'is_trained', False)
        for model_name, model in models.items()
    }
    
    return HealthResponse(
        status="healthy" if any(models_status.values()) else "unhealthy",
        models_loaded=models_status,
        timestamp=datetime.now().isoformat()
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict_fraud(request: PredictionRequest):
    """Make fraud predictions for transactions."""
    start_time = datetime.now()
    
    # Validate model type
    if request.model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available: {list(models.keys())}"
        )
    
    model = models[request.model_type]
    if model is None or not getattr(model, 'is_trained', False):
        raise HTTPException(
            status_code=503,
            detail=f"Model {request.model_type} is not loaded or trained"
        )
    
    try:
        # Prepare data
        df = prepare_dataframe(request.transactions)
        
        # Make predictions
        with PREDICTION_LATENCY.labels(model=request.model_type).time():
            probabilities = model.predict_proba(df)
            predictions = (probabilities >= request.threshold).astype(int)
        
        # Format results
        results = []
        for i, (prob, pred) in enumerate(zip(probabilities, predictions)):
            result = {
                "customer_id": request.transactions[i].customer_id,
                "fraud_probability": float(prob),
                "is_fraud": int(pred),
                "confidence": float(abs(prob - 0.5) * 2),  # Distance from decision boundary
                "transaction_index": i
            }
            results.append(result)
            
            # Update metrics
            PREDICTION_COUNTER.labels(
                model=request.model_type, 
                prediction=str(pred)
            ).inc()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return PredictionResponse(
            predictions=results,
            model_used=request.model_type,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/batch")
async def predict_batch(request: BatchPredictionRequest):
    """Batch prediction endpoint for large datasets."""
    start_time = datetime.now()
    
    if request.model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available: {list(models.keys())}"
        )
    
    model = models[request.model_type]
    if model is None or not getattr(model, 'is_trained', False):
        raise HTTPException(
            status_code=503,
            detail=f"Model {request.model_type} is not loaded or trained"
        )
    
    try:
        # Convert JSON data to DataFrame
        df = pd.DataFrame(request.data)
        
        # Make predictions
        with PREDICTION_LATENCY.labels(model=request.model_type).time():
            probabilities = model.predict_proba(df)
            predictions = (probabilities >= 0.5).astype(int)
        
        # Create results
        results = {
            "predictions": predictions.tolist(),
            "probabilities": probabilities.tolist(),
            "fraud_count": int(predictions.sum()),
            "total_count": len(predictions),
            "fraud_rate": float(predictions.mean())
        }
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return {
            "results": results,
            "model_used": request.model_type,
            "processing_time": processing_time,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")


@app.get("/models")
async def list_models():
    """List available models and their status."""
    model_info = {}
    for name, model in models.items():
        if model is not None:
            model_info[name] = {
                "loaded": True,
                "trained": getattr(model, 'is_trained', False),
                "type": type(model).__name__
            }
        else:
            model_info[name] = {
                "loaded": False,
                "trained": False,
                "type": None
            }
    
    return model_info


@app.post("/models/{model_type}/reload")
async def reload_model(model_type: str):
    """Reload a specific model."""
    if model_type not in models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model type. Available: {list(models.keys())}"
        )
    
    try:
        await load_models()  # This will reload all models
        return {"message": f"Model {model_type} reloaded successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to reload model: {str(e)}")


@app.get("/metrics")
async def get_metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/stats")
async def get_stats():
    """Get API usage statistics."""
    # This would typically pull from a database or monitoring system
    # For now, return basic info
    return {
        "total_predictions": "See /metrics endpoint",
        "models_active": len([m for m in models.values() if m is not None]),
        "uptime": "See /metrics endpoint",
        "timestamp": datetime.now().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)