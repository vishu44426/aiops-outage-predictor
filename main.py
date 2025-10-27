# main.py
import os
import sys
import logging
from datetime import datetime, timedelta

# Add the source directory to the system path 
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

try:
    # 1. Import Logging Setup and Configuration
    from outage_predictor.utils import config, setup_logging 
  
    # 2. Import Data Preparation Functions
    from outage_predictor.data.synthetic_generator import generate_synthetic_logs, ingest_to_sqlite
    from scripts.build_vector_db import run_rag_data_prep
    from scripts.fine_tune_embeddings import run_embedding_finetune

    # Import the Fine-Tuning function
    from scripts.fine_tune_embeddings import run_embedding_finetune

    # 3. Import Training Function
    from scripts.generate_features import run_feature_generation
    from outage_predictor.models.train import train_model
    
    # 4. Import Deployment/Simulation Functions
    from outage_predictor.deployment.mlflow_utils import transition_model_to_production
    
    # Import the main prediction function AND the new reloader function
    from outage_predictor.deployment.predictor_service import predict_realtime, force_reload_model
    
    import pandas as pd 
    
except ImportError as e:
    print(f"FATAL ERROR: Failed to import necessary modules: {e}")
    print("Please ensure you have run 'pip install -r requirements.txt' and are running from the project root directory.")
    sys.exit(1)

logger = logging.getLogger('outage_predictor')

def step_1a_data_preparation():
    # ... (this function is unchanged)
    logger.info("=" * 70)
    logger.info("STEP 1: Starting Data Generation and Ingestion (01_prepare_data.py logic)")
    logger.info("=" * 70)
    C = config['data']
    logger.info("Generating synthetic log data...")
    log_data_df = generate_synthetic_logs()
    logger.info(f"Ingesting {len(log_data_df)} samples to SQLite...")
    ingest_to_sqlite(log_data_df)
    outage_count = log_data_df['Outage_Flag'].sum()
    logger.info(f"Total Samples Generated: {len(log_data_df)}")
    logger.info(f"Total Outage Samples (Flag=1) generated: {outage_count} (Approx {outage_count/len(log_data_df)*100:.2f}%)")
    logger.info(f"Data saved to: {C.get('DB_PATH', 'data/log_outage_data.db')}")
    logger.info("--- Step 1 Complete ---\n")

def step_1b_build_vector_db():
    """Builds the RAG vector database from the new SQLite data."""
    logger.info("=" * 70)
    logger.info("STEP 1B: Starting RAG Vector Database Build")
    logger.info("=" * 70)
    
    try:
        run_rag_data_prep(use_finetuned=False, suffix=config['mlops'].get('RAG_DB_SUFFIX_BASE'))
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: RAG Vector DB build failed.", exc_info=True)
        sys.exit(1)
        
    logger.info("--- Step 1B Complete ---\n")

# Added new step for fine-tuning
def step_1c_finetune_embeddings():
    """Fine-tunes the embedding model on the new data."""
    logger.info("=" * 70)
    logger.info("STEP 1C: Starting Embedding Model Fine-Tuning")
    logger.info("=" * 70)
    
    try:
        run_embedding_finetune()
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: Embedding fine-tuning failed.", exc_info=True)
        sys.exit(1)
        
    logger.info("--- Step 1C Complete ---\n")

def step_1d_build_vector_db_finetuned():
    """Builds the RAG vector database using the FINE-TUNED embedding model."""
    logger.info("=" * 70)
    logger.info("STEP 1D: Building FINE-TUNED RAG Vector DB")
    logger.info("=" * 70)
    try:
        # Call with use_finetuned=True and fine-tuned suffix
        run_rag_data_prep(use_finetuned=True, suffix=config['mlops'].get('RAG_DB_SUFFIX_FINETUNED'))
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: FINE-TUNED RAG DB build failed.", exc_info=True)
        sys.exit(1)
    logger.info("--- Step 1D Complete ---\n")

def step_1e_generate_features():
    """Generates features and saves them to the feature store."""
    logger.info("=" * 70)
    logger.info("STEP 1E: Generating and Storing Features")
    logger.info("=" * 70)
    try:
        run_feature_generation()
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: Feature generation failed.", exc_info=True)
        sys.exit(1)
    logger.info("--- Step 1E Complete ---\n")

def step_2_model_training():
    # ... (this function is unchanged)
    logger.info("=" * 70)
    logger.info("STEP 2: Starting Model Training and MLflow Tracking (02_run_training.py logic)")
    logger.info("=" * 70)
    logger.info("NOTE: Please ensure the MLflow Tracking Server is running at http://127.0.0.1:5000")
    try:
        train_model()
    except Exception as e:
        logger.error(f"\nCRITICAL ERROR: Training failed. Please ensure the SQLite DB is populated (Step 1) and the MLflow Tracking Server is accessible.", exc_info=True)
        sys.exit(1)
    logger.info("--- Step 2 Complete ---\n")


def step_3_deployment_simulation():
    """Transitions the model and simulates a real-time prediction request."""
    logger.info("=" * 70)
    logger.info("STEP 3: Starting Deployment Management and Inference Simulation (03_simulate_inference.py logic)")
    logger.info("=" * 70)

    # 3a. Transition the latest trained model version to Production
    logger.info("Transitioning latest model version to 'Production' alias...")
    transition_model_to_production() 

    # Force the service to reload the model from MLflow
    # This ensures it picks up the *new* production model from step 3a
    logger.info("Forcing predictor service to reload the new production model...")
    force_reload_model()

    # 3b. Simulate Real-Time Request (testing the new production model)
    logger.info("\nSimulating Real-Time Inference Request...")
    
    current_time = datetime.now()
    recent_logs = [
        {
            "Timestamp": (current_time - timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": "Network",
            "Severity": "INFO",
            "Message": "Packet received"
        },
        {
            "Timestamp": (current_time - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": "CPU",
            "Severity": "WARNING",
            "Message": "CPU load at 70%"
        },
        {
            "Timestamp": (current_time - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": "CPU",
            "Severity": "ERROR",
            "Message": "CPU temperature threshold warning"
        },
        {
            "Timestamp": (current_time - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": "CPU",
            "Severity": "ERROR",
            "Message": "CPU fan speed low"
        },
        {
            "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Component": "CPU",
            "Severity": "CRITICAL",
            "Message": "CPU OVERHEAT SHUTDOWN IMMINENT"
        }
    ]
    
    result = predict_realtime(recent_logs)
    
    logger.info("-" * 50)
    if "error" in result:
        logger.error(f"Prediction Error: {result['error']}")
    else:
        logger.info(f"Prediction Time: {result['prediction_timestamp']}")
        logger.info(f"Correlation ID (for monitoring): {result['correlationid']}")
        logger.info(f"Outage Probability: {result['prediction_probability']:.4f}")
        logger.info(f"Alert Status: {result['alert_status']}")
    logger.info("-" * 50)
    
    logger.info("--- Step 3 Complete ---\n")


if __name__ == '__main__':
    # ... (this function is unchanged)
    logger.info("--- Starting End-to-End MLOps Playbook Execution ---")
    
    # 1. Create the data
    #step_1a_data_preparation()

    # 2. Build the BASE vector DB
    #step_1b_build_vector_db()

    # 3. Train the embedding model and register it
    #step_1c_finetune_embeddings()

    # 4. Build the FINE-TUNED vector DB
    #step_1d_build_vector_db_finetuned()

    # 5. Generate and store features
    #step_1e_generate_features()
    
    # 6. Train the classic ML model and register it
    #step_2_model_training()

    # 7. Test the classic ML model
    step_3_deployment_simulation()
    
    logger.info("--- All Sequential Steps Completed Successfully ---")