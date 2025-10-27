# src/outage_predictor/data/data_loader.py

import pandas as pd
from sqlalchemy import create_engine
import logging

from..utils import config

logger = logging.getLogger('outage_predictor')
FEATURE_TABLE_NAME = "engineered_features" # Use the same name as in generate_features.py

def load_data_from_db():
    """Loads all data from the SQLite database."""
    C = config['data']
    db_path = C.get('DB_PATH', "data/log_outage_data.db")
    table_name = C.get('TABLE_NAME', "server_logs")
    
    logger.info(f"Loading data from {db_path}...")
    
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        
        query = f"SELECT * FROM {table_name} ORDER BY Timestamp ASC"
        
        # Explicitly parse Timestamp column on load
        df = pd.read_sql(query, engine, parse_dates=['Timestamp'])
        
        # Ensure Timestamp is the index and correctly sorted for time-series operations
        df = df.set_index('Timestamp').sort_index()
        logger.info(f"Successfully loaded {len(df)} records.")
        return df
    except Exception as e:
        logger.error(f"Failed to load data from database: {e}")
        raise

def chronological_split(X, y, train_ratio, val_ratio):
    """
    Splits the data chronologically into Training, Validation, and Test sets.
    CRITICAL for time-series to avoid data leakage.[1]
    """
    logger.info("Applying chronological split to data...")
    total_samples = len(X)
    
    train_end = int(total_samples * train_ratio)
    val_end = int(total_samples * (train_ratio + val_ratio))
    
    X_train, y_train = X[:train_end], y[:train_end]
    X_val, y_val = X[train_end:val_end], y[train_end:val_end]
    X_test, y_test = X[val_end:], y[val_end:]
    
    logger.info(f"Split sizes: Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}")
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def get_chronological_splits():
    """
    [ENHANCEMENT] Wrapper to load pre-computed features and target
    from the feature store table and apply chronological split.
    """
    C = config['data']
    db_path = C.get('DB_PATH', "data/log_outage_data.db")

    logger.info(f"Loading features from table '{FEATURE_TABLE_NAME}' in {db_path}...")
    try:
        engine = create_engine(f"sqlite:///{db_path}")
        query = f"SELECT * FROM {FEATURE_TABLE_NAME} ORDER BY Timestamp ASC"
        # Load features, ensuring Timestamp is parsed and set as index
        df_features = pd.read_sql(query, engine, index_col='Timestamp', parse_dates=['Timestamp'])
        logger.info(f"Successfully loaded {len(df_features)} records with features.")

        # Separate features (X) and target (y)
        y = df_features['Outage_Flag']
        X = df_features.drop(columns=['Outage_Flag']) # Keep all engineered features

        # Apply chronological split
        C_SPLIT = config['data'] # Use data config for ratios
        return chronological_split(X, y, C_SPLIT.get('TRAIN_RATIO', 0.7), C_SPLIT.get('VAL_RATIO', 0.15))

    except Exception as e:
        logger.error(f"Failed to load features from table '{FEATURE_TABLE_NAME}': {e}")
        logger.error("Did you run the feature generation step in main.py?")
        raise