# src/outage_predictor/deployment/mlflow_utils.py

import mlflow
from mlflow.tracking import MlflowClient
import logging

from..utils import config

logger = logging.getLogger('outage_predictor')

def transition_model_to_production(version=None):
    """
    Sets the 'Production' alias for the latest or specified
    model version in the MLflow Model Registry.
    """
    C_MLOPS = config['mlops']
    MODEL_NAME = C_MLOPS.get('MODEL_NAME', "Outage_Predictor_GBM")
    
    mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI')) 
    
    client = MlflowClient()
    
    if version is None:
        logger.info(f"Fetching latest model version for '{MODEL_NAME}'...")
        try:
            # Get latest versions (no stage specified)
            latest_versions = client.get_latest_versions(MODEL_NAME)
            
            if not latest_versions:
                logger.error(f"No model versions found for '{MODEL_NAME}' at all.")
                return
            
            version = latest_versions[0].version 
            logger.info(f"Latest version found: {version}")
        except Exception as e:
             logger.error(f"Could not find model '{MODEL_NAME}' in Registry: {e}. Ensure training script ran successfully.")
             return
        
    try:
        # Use set_registered_model_alias instead of transition_model_version_stage
        logger.info(f"Setting 'Production' alias for model '{MODEL_NAME}' Version {version}...")
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="Production",
            version=version
        )
        logger.info(f"Model '{MODEL_NAME}' Version {version} successfully set as 'Production' alias.")
    except Exception as e:
        logger.error(f"Error setting model alias for version {version}: {e}")