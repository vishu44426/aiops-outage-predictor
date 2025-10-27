# src/outage_predictor/deployment/predictor_service.py

import mlflow
import mlflow.sklearn 
import pandas as pd
import uuid
import logging
from typing import List, Dict, Any
from datetime import datetime

from..utils import config

logger = logging.getLogger('outage_predictor')

C_MLOPS = config.get('mlops', {})
MLFLOW_URI = C_MLOPS.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5000')
MODEL_NAME = C_MLOPS.get('MODEL_NAME', 'Outage_Predictor_GBM')
ALERT_THRESHOLD = C_MLOPS.get('ALERT_THRESHOLD', 0.5)

MODEL_URI = f"models:/{MODEL_NAME}@Production"

production_model = None

def load_production_model():
    """
    Loads the model on-demand and stores it globally.
    """
    global production_model
    
    if production_model is None:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            logger.info(f"Attempting to load production model from {MODEL_URI} as scikit-learn model...")
            
            # Load as a scikit-learn model to get .predict_proba()
            production_model = mlflow.sklearn.load_model(MODEL_URI)
            
            logger.info("Production scikit-learn model loaded successfully.")
        except Exception as e:
            logger.error(f"Error loading production model from {MODEL_URI}: {e}")
            logger.error(f"MLflow URI was set to: {MLFLOW_URI}")
            logger.error("Inference service will not be operational.")
            production_model = "ERROR" 


def prepare_production_payload(log_data_list: List[Dict[str, Any]]):
    """
    Prepares raw log data into a DataFrame format expected by the Feature Extractor.
    """
    correlation_id = str(uuid.uuid4())
    input_df = pd.DataFrame(log_data_list)
    
    if 'Timestamp' not in input_df.columns:
        raise ValueError("Input data must contain a 'Timestamp' column.")
    
    try:
        input_df['Timestamp'] = pd.to_datetime(input_df['Timestamp']) 
    except Exception as e:
        logger.error(f"Timestamp conversion failed: {e}")
        raise ValueError("Input data must contain valid 'Timestamp' strings.")

    prediction_timestamp = input_df['Timestamp'].max()
    
    logger.debug(f"Prepared payload with correlationid: {correlation_id}")
    
    return correlation_id, input_df, prediction_timestamp

def predict_realtime(log_data_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Simulates a real-time prediction request using the deployed pipeline.
    """
    load_production_model()

    if production_model is None or production_model == "ERROR":
        logger.error("Prediction failed: Model is not loaded or failed to load.")
        return {"error": "Model not loaded.", "correlationid": str(uuid.uuid4())}
    
    try:
        correlation_id, X_raw_input, prediction_timestamp = prepare_production_payload(log_data_list)
    except Exception as e:
        logger.error(f"Prediction failed due to payload error: {e}")
        return {"error": f"Invalid input payload: {e}", "correlationid": str(uuid.uuid4())}

    try:
        # Call .predict_proba() on the loaded sklearn pipeline
        y_pred_proba_all = production_model.predict_proba(X_raw_input) 
        
        # Get the last prediction in the window
        y_pred_proba_last = y_pred_proba_all[-1]
        
        # Probability of the positive class (1=Outage) is at index 1
        outage_probability = y_pred_proba_last[1] 
        
        alert_status = "ALERT: Outage Imminent" if outage_probability > ALERT_THRESHOLD else "Normal Operations"
        
        logger.info(f"Prediction result: Probability={outage_probability:.4f}, Alert={alert_status}")
        
        return {
            "prediction_probability": float(outage_probability),
            "alert_status": alert_status,
            "correlationid": correlation_id,
            "prediction_timestamp": prediction_timestamp.isoformat()
        }
        
    except Exception as e:
        logger.error(f"Prediction inference failed: {e}", exc_info=True)
        return {"error": f"Prediction failed during inference: {e}", "correlationid": correlation_id}


def force_reload_model():
    """
    Resets the global model variable.
    """
    global production_model
    if production_model is not None:
        logger.info("Forcing model reload by setting global model cache to None.")
        production_model = None
    else:
        logger.info("Model is already None. No reload necessary.")