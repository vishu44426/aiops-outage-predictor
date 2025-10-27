# src/outage_predictor/utils.py

import yaml
import os
import logging
from logging.config import dictConfig # For structured logging setup

# --- Logging Configuration ---
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'default': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
    },
    'handlers': {
        'console': {
            'class': 'logging.StreamHandler',
            'formatter': 'default',
            'level': 'INFO',
            'stream': 'ext://sys.stdout',
        },
    },
    'root': {
        'handlers': ['console'],
        'level': 'INFO',
    },
    'loggers': {
        'outage_predictor': { 
            'handlers': ['console'],
            'level': 'INFO',
            'propagate': False,
        },
    }
}

def setup_logging():
    """Applies the default logging configuration."""
    dictConfig(LOGGING_CONFIG)

# --- Configuration Loader ---

# Global logger instance for configuration module
logger = logging.getLogger('outage_predictor')

def load_config(config_path="config"):
    """Loads all configuration files from the config directory."""
    config = {}
    
    # Calculate project root path relative to this file
    base_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.join(base_dir, '../../')
    
    data_path = os.path.join(project_root, config_path, "config_data.yaml")
    model_path = os.path.join(project_root, config_path, "config_model.yaml")
    mlops_path = os.path.join(project_root, config_path, "config_mlops.yaml")

    try:
      with open(data_path, 'r') as f:
          config['data'] = yaml.safe_load(f).get('DATA', {})

      with open(model_path, 'r') as f:
          config['model'] = yaml.safe_load(f)

      # Load and merge MLOps config
      with open(mlops_path, 'r') as f:
          config['mlops'] = yaml.safe_load(f)
          
      logger.info("Configuration files loaded successfully.")
      
    except FileNotFoundError as e:
        logger.error(f"Error loading config: {e}. Ensure config files exist.")
        
        # Added complete lists for FAILURE_COMPONENTS and SEVERITIES to resolve errors.
        config['data'] = {'DB_PATH': "data/log_outage_data.db", 'TABLE_NAME': "server_logs", 
                          'TOTAL_SAMPLES': 10000, 'START_TIME': "2024-01-01 00:00:00", 
                          'LOG_RATE_PER_MIN': 1.0, 
                          'FAILURE_COMPONENTS': ['database', 'network', 'api', 'storage'], 
                          'SEVERITIES': ['INFO', 'WARNING', 'ERROR', 'CRITICAL'], 
                          'SEVERITY_PROBS': [0.9, 0.07, 0.025, 0.005], 'NUM_OUTAGES': 5, 
                          'PRECURSOR_WINDOW_MINUTES': 15, 'PRECURSOR_ERROR_RATE': 0.7, 
                          'ROLLING_WINDOW_MINUTES': 30, 'TRAIN_RATIO': 0.7, 'VAL_RATIO': 0.15, 'TEST_RATIO': 0.15}
        config['model'] = {'MODEL_NAME': "Outage_Predictor_GBM", 'XGBOOST_PARAMS': {'objective': 'binary:logistic', 'n_estimators': 10, 'max_depth': 3}}
        # Added fallback for MLOps config
        config['mlops'] = {'MLFLOW_TRACKING_URI': 'http://127.0.0.1:5000', 'MODEL_NAME': 'Outage_Predictor_GBM', 'ALERT_THRESHOLD': 0.5}
        
        logger.warning("Using default configurations due to FileNotFoundError.")
        
    return config

# Initialize logging and load config upon module import
setup_logging()
config = load_config()