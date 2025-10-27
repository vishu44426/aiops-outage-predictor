import os
import sys
import logging
import pandas as pd
from sqlalchemy import create_engine

# Setup path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from outage_predictor.utils import config, setup_logging
    from outage_predictor.data.data_loader import load_data_from_db # Use the existing loader
    from outage_predictor.features.build_features import apply_feature_engineering # Use existing logic

except ImportError as e:
    print(f"FATAL ERROR: Could not import project modules: {e}")
    sys.exit(1)

# --- Initialize Logging ---
setup_logging()
logger = logging.getLogger('outage_predictor_feature_gen')

# --- Configuration ---
C_DATA = config['data']
DB_PATH = C_DATA.get('DB_PATH', 'data/log_outage_data.db')
FEATURE_TABLE_NAME = "engineered_features" # New table name

def run_feature_generation():
    """Loads raw data, applies feature engineering, and saves to a new DB table."""
    logger.info("=" * 70)
    logger.info("STEP: Starting Feature Generation")
    logger.info("=" * 70)

    try:
        # 1. Load raw data (including Outage_Flag)
        logger.info("Loading raw data from database...")
        # Make load_data_from_db return the full raw df for this step
        # You might need to adjust load_data_from_db or create a variant
        engine = create_engine(f"sqlite:///{DB_PATH}")
        query = f"SELECT * FROM {C_DATA.get('TABLE_NAME', 'server_logs')} ORDER BY Timestamp ASC"
        df_raw_full = pd.read_sql(query, engine, index_col='Timestamp', parse_dates=['Timestamp'])
        logger.info(f"Loaded {len(df_raw_full)} raw log records.")

        # Separate features input (X) from target (y)
        y = df_raw_full['Outage_Flag']
        X_raw = df_raw_full.drop(columns=['Outage_Flag', 'Message']).copy() # Keep Component, Severity etc.

        # 2. Apply feature engineering
        logger.info("Applying feature engineering logic...")
        df_features = apply_feature_engineering(X_raw)
        logger.info(f"Generated {len(df_features.columns)} features for {len(df_features)} timestamps.")

        # 3. Combine features with the target variable and timestamp index
        df_feature_store = pd.concat([df_features, y], axis=1)
        # Drop rows where features couldn't be calculated (initial window NaNs)
        df_feature_store.dropna(subset=df_features.columns, inplace=True) # Important!
        df_feature_store.reset_index(inplace=True) # Make Timestamp a column for SQL

        # 4. Save to the new table in SQLite
        logger.info(f"Saving features to table '{FEATURE_TABLE_NAME}' in {DB_PATH}...")
        df_feature_store.to_sql(FEATURE_TABLE_NAME, engine, if_exists='replace', index=False)

        logger.info(f"Successfully saved {len(df_feature_store)} records with features.")
        logger.info("--- Feature Generation Complete ---")

    except Exception as e:
        logger.error(f"FATAL ERROR during feature generation: {e}", exc_info=True)
        sys.exit(1)

if __name__ == "__main__":
    run_feature_generation()