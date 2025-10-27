# app.py

import uvicorn
from fastapi import FastAPI, HTTPException, Body
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import sys
import os
import logging
import uuid 

from sqlalchemy import create_engine # Add sqlalchemy
import pandas as pd # Add pandas
from fastapi import Query # To accept query parameters
from outage_predictor.utils import setup_logging

# --- Initialize Logging and App ---
setup_logging()
logger = logging.getLogger('outage_predictor_api')

try:
    from outage_predictor.utils import config # Import config here

    # --- Classic ML Imports ---
    from outage_predictor.deployment.predictor_service import predict_realtime, load_production_model, production_model
    
    # --- LLM/RAG Imports ---
    from outage_predictor.deployment.llm_predictor_service import (
        predict_realtime_rag_base, load_vector_store_base,
        predict_realtime_rag_finetuned, load_vector_store_finetuned
    )
    
    from outage_predictor.utils import setup_logging
except ImportError as e:
    print(f"FATAL ERROR: Could not import project modules: {e}")
    sys.exit(1)


# --- Configuration (AFTER importing config) ---
C_DATA = config.get('data', {})
DB_PATH = C_DATA.get('DB_PATH', "data/log_outage_data.db")
FEATURE_TABLE_NAME = "engineered_features"

app = FastAPI(
    title="AIOps Outage Predictor API",
    description="API for predicting server outages. Includes Classic ML and RAG models.",
    version="1.0.0"
)

# --- API Data Models ---
class LogEntry(BaseModel):
    Timestamp: str
    Component: str
    Severity: str
    Message: str

# Add similarity_score
class RelatedEvent(BaseModel):
    log_snippet: str
    end_timestamp: str
    was_outage_flag: int
    similarity_score: Optional[float] = None # Add score

class PredictionResponse(BaseModel):
    """Defines the structure of the prediction response."""
    prediction_probability: Optional[float] = None
    alert_status: Optional[str] = None
    correlationid: str
    prediction_timestamp: Optional[str] = None
    error: Optional[str] = None
    # Add explanation field
    explanation: Optional[str] = None
    related_events: Optional[List[RelatedEvent]] = None

# ... (example_log_payload, startup, and root endpoints are unchanged) ...
example_log_payload = [
    {"Timestamp": "2025-10-27 12:02:00", "Component": "CPU", "Severity": "WARNING", "Message": "CPU load at 70%"},
    {"Timestamp": "2025-10-27 12:03:00", "Component": "CPU", "Severity": "ERROR", "Message": "CPU temperature threshold warning"},
    {"Timestamp": "2025-10-27 12:04:00", "Component": "CPU", "Severity": "ERROR", "Message": "CPU fan speed low"},
    {"Timestamp": "2025-10-27 12:05:00", "Component": "CPU", "Severity": "CRITICAL", "Message": "CPU OVERHEAT SHUTDOWN IMMINENT"}
]

@app.on_event("startup")
def startup_load_models():
    """At application startup, trigger loaders for all models/stores."""
    logger.info("Application startup: triggering model/store loads...")
    load_production_model() # Classic ML
    load_vector_store_base() # Base RAG Store
    load_vector_store_finetuned() # Fine-tuned RAG Store

@app.get("/")
def read_root():
    is_loaded = (production_model is not None and production_model != "ERROR")
    if not is_loaded:
        load_production_model()
        is_loaded = (production_model is not None and production_model != "ERROR")
    return {
        "message": "AIOps Predictor API is running.",
        "model_loaded": is_loaded
    }

# --- ENDPOINT 1: CLASSIC ML (XGBOOST) ---
@app.post("/predict-classic", response_model=PredictionResponse)
def post_predict_classic(log_entries: List[LogEntry] = Body(..., example=example_log_payload)):
    """Predicts using the Classic ML (XGBoost) model."""
    # ... (Unchanged)
    try:
        log_data_list = [entry.model_dump() for entry in log_entries]
        result = predict_realtime(log_data_list)
        if "error" in result and result["error"]: return result
        return result
    except Exception as e: return PredictionResponse(correlationid=str(uuid.uuid4()), error=f"Internal server error: {e}")

# --- ENDPOINT 2: LLM (RAG - BASE MODEL) ---
@app.post("/predict-llm-rag-base", response_model=PredictionResponse)
def post_predict_llm_rag_base(log_entries: List[LogEntry] = Body(..., example=example_log_payload)):
    """Predicts using the RAG method with the BASE embedding model."""
    try:
        log_data_list = [entry.model_dump() for entry in log_entries]
        # Call the specific base prediction function
        result = predict_realtime_rag_base(log_data_list, k_neighbors=5)
        if "error" in result and result["error"]: return result
        return result
    except Exception as e: return PredictionResponse(correlationid=str(uuid.uuid4()), error=f"Internal server error: {e}")

# --- ENDPOINT 3: LLM (RAG - FINE-TUNED MODEL) ---
@app.post("/predict-llm-rag-finetuned", response_model=PredictionResponse)
def post_predict_llm_rag_finetuned(log_entries: List[LogEntry] = Body(..., example=example_log_payload)):
    """Predicts using the RAG method with the FINE-TUNED embedding model."""
    try:
        log_data_list = [entry.model_dump() for entry in log_entries]
        # Call the specific fine-tuned prediction function
        result = predict_realtime_rag_finetuned(log_data_list, k_neighbors=5)
        if "error" in result and result["error"]: return result
        return result
    except Exception as e: return PredictionResponse(correlationid=str(uuid.uuid4()), error=f"Internal server error: {e}")

# --- ENDPOINT 4: Endpoint to view features
@app.get("/features/latest", response_model=List[Dict[str, Any]])
def get_latest_features(limit: int = Query(default=5, ge=1, le=100)):
    """Retrieves the latest 'limit' feature vectors from the feature store."""
    logger.info(f"Received request for latest {limit} features.")
    try:
        engine = create_engine(f"sqlite:///{DB_PATH}")
        # Query the latest features by timestamp
        query = f"SELECT * FROM {FEATURE_TABLE_NAME} ORDER BY Timestamp DESC LIMIT {limit}"
        df_features = pd.read_sql(query, engine)

        # Convert to list of dictionaries for JSON response
        features_list = df_features.to_dict(orient='records')
        return features_list

    except Exception as e:
        logger.error(f"Error fetching latest features: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve features: {e}")

# --- ENDPOINT 4: Endpoint to view features at a specific timestamp
@app.get("/features/{timestamp_str}", response_model=Dict[str, Any])
def get_features_at_timestamp(timestamp_str: str):
    """Retrieves the feature vector for a specific timestamp (ISO format)."""
    logger.info(f"Received request for features at timestamp: {timestamp_str}")
    try:
        # Validate timestamp format (optional but recommended)
        try:
            target_ts = pd.to_datetime(timestamp_str).isoformat(sep=' ', timespec='seconds')
        except ValueError:
            raise HTTPException(status_code=400, detail="Invalid timestamp format. Use YYYY-MM-DD HH:MM:SS or ISO format.")

        engine = create_engine(f"sqlite:///{DB_PATH}")
        # Query features for the specific timestamp
        query = f"SELECT * FROM {FEATURE_TABLE_NAME} WHERE Timestamp = ?"
        df_feature = pd.read_sql(query, engine, params=(target_ts,))

        if df_feature.empty:
            raise HTTPException(status_code=404, detail=f"No features found for timestamp {target_ts}")

        # Convert the single row DataFrame to a dictionary
        feature_dict = df_feature.iloc[0].to_dict()
        return feature_dict

    except HTTPException: # Re-raise FastAPI exceptions
        raise
    except Exception as e:
        logger.error(f"Error fetching features for {timestamp_str}: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve features: {e}")
    
# --- Run the Server ---
if __name__ == "__main__":
    logger.info("Starting FastAPI server on http://127.0.0.1:8000")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)