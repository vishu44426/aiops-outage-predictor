# scripts/fine_tune_embeddings.py

import os
import sys
import logging
import pandas as pd
from typing import List, Dict
import random
import mlflow
import mlflow.sentence_transformers
from mlflow.tracking import MlflowClient

# Setup path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))

try:
    from outage_predictor.utils import config, setup_logging
    from outage_predictor.data.data_loader import load_data_from_db

    from sentence_transformers import SentenceTransformer, losses, models
    from sentence_transformers.readers import InputExample
    from torch.utils.data import DataLoader

except ImportError as e:
    print(f"FATAL ERROR: Could not import project modules or RAG libraries: {e}")
    print("Please ensure you have run 'pip install -r requirements.txt'")
    sys.exit(1)

# --- Initialize Logging ---
setup_logging()
logger = logging.getLogger('outage_predictor_finetune')

# --- Configuration ---
C_DATA = config['data']
C_MLOPS = config['mlops']
BASE_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_MODEL_NAME = C_MLOPS.get('EMBEDDING_MODEL_REGISTRY_NAME', "AIOps_Embedding_Model")
BATCH_SIZE = 16
EPOCHS = 1

def create_finetuning_dataset(df: pd.DataFrame) -> List[InputExample]:
    # ... (This function is unchanged - make sure it includes Message)
    logger.info("Creating fine-tuning dataset...")
    df['outage_group'] = (df['Outage_Flag'].diff() == 1).cumsum()
    outage_logs = df[df['Outage_Flag'] == 1]
    normal_logs = df[df['Outage_Flag'] == 0]
    train_examples = []

    logger.info("Generating positive pairs for outage events...")
    outage_groups = outage_logs.groupby('outage_group')
    for name, group in outage_groups:
        if len(group) < 2: continue
        log_texts = ("Severity: " + group['Severity'] + " Component: " + group['Component'] + " Message: " + group['Message']).tolist()
        for i in range(len(log_texts)):
            for j in range(i + 1, len(log_texts)):
                text_i = str(log_texts[i]) if pd.notna(log_texts[i]) else ""
                text_j = str(log_texts[j]) if pd.notna(log_texts[j]) else ""
                if text_i and text_j: train_examples.append(InputExample(texts=[text_i, text_j], label=1.0))

    logger.info("Generating positive pairs for normal operations...")
    normal_texts = ("Severity: " + normal_logs['Severity'] + " Component: " + normal_logs['Component'] + " Message: " + normal_logs['Message']).tolist()
    normal_texts = [str(text) for text in normal_texts if pd.notna(text)]

    if not normal_texts or not train_examples:
        logger.warning("Not enough data to create normal pairs or outage pairs. Skipping normal pairs.")
    else:
        num_normal_pairs_to_create = min(len(train_examples), (len(normal_texts) * (len(normal_texts)-1)) // 2)
        if num_normal_pairs_to_create > 0 and len(normal_texts) >= 2:
            for _ in range(num_normal_pairs_to_create):
                 try:
                    pair = random.sample(normal_texts, 2)
                    train_examples.append(InputExample(texts=[pair[0], pair[1]], label=1.0))
                 except ValueError:
                     logger.warning(f"Could not sample 2 unique normal texts from a list of size {len(normal_texts)}. Stopping normal pair generation.")
                     break
        else:
             logger.warning("Not enough unique normal texts to create pairs.")

    logger.info(f"Created {len(train_examples)} total positive pairs for training.")
    return train_examples


def run_embedding_finetune():
    """Main function to run the embedding model fine-tuning pipeline."""
    logger.info("=" * 70)
    logger.info("STEP 1C: Starting Embedding Model Fine-Tuning")
    logger.info("=" * 70)

    try:
        # 1. Setup MLflow
        logger.info("Setting up MLflow tracking for fine-tuning...")
        mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI'))
        mlflow.set_experiment("/AIOps_Outage_Prediction")

        # 2. Load data & Create training dataset
        df_raw = load_data_from_db()
        train_examples = create_finetuning_dataset(df_raw)

        if not train_examples:
            logger.error("No training examples were created. Aborting.")
            return

        # 3. Load base model
        logger.info(f"Loading base model: {BASE_MODEL_NAME}...")
        model = SentenceTransformer(BASE_MODEL_NAME)

        # 4. Dataloader and loss
        train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
        train_loss = losses.MultipleNegativesRankingLoss(model)

        # 5. Start an MLflow run to track the tuning
        with mlflow.start_run(run_name="Finetune_Embedding_Model"):
            mlflow.log_param("base_model", BASE_MODEL_NAME)
            mlflow.log_param("num_examples", len(train_examples))
            mlflow.log_param("epochs", EPOCHS)

            # 6. Fine-tune the model
            logger.info(f"Starting model fine-tuning for {EPOCHS} epoch(s)...")
            # We don't need output_path here as we log directly to MLflow
            model.fit(
                train_objectives=[(train_dataloader, train_loss)],
                epochs=EPOCHS,
                show_progress_bar=True,
                # output_path=FINE_TUNED_MODEL_PATH, # Not needed when logging
                # save_best_model=False
            )

            # 7. Log and register the model
            logger.info(f"Training complete. Logging and registering model as '{EMBEDDING_MODEL_NAME}'...")

            sample_input = ["Severity: ERROR Component: CPU Message: Test message"]
            signature = mlflow.models.infer_signature(sample_input, model.encode(sample_input))

            # [FIX] Use the correct keyword argument 'model' instead of 'sentence_transformer_model'
            mlflow.sentence_transformers.log_model(
                model=model,
                artifact_path="embedding_model",
                signature=signature,
                registered_model_name=EMBEDDING_MODEL_NAME
            )

            logger.info("Model logged to MLflow successfully.")

        # 8. Promote the new model to "Production"
        logger.info(f"Promoting latest version of '{EMBEDDING_MODEL_NAME}' to Production alias...")
        client = MlflowClient()
        # Ensure MLflow URI is set for the client
        client._tracking_client.tracking_uri = C_MLOPS.get('MLFLOW_TRACKING_URI')

        # Get the latest version from the run we just finished
        # Note: This assumes no other runs are happening concurrently for this model
        latest_versions = client.get_latest_versions(EMBEDDING_MODEL_NAME, stages=["None"])
        if not latest_versions:
             logger.warning(f"Could not find a new version for {EMBEDDING_MODEL_NAME}. Trying Staging/Production.")
             latest_versions = client.get_latest_versions(EMBEDDING_MODEL_NAME, stages=["Staging", "Production"])

        if not latest_versions:
             logger.error(f"FATAL: Could not find any version for model {EMBEDDING_MODEL_NAME} to promote.")
             raise ValueError(f"No versions found for {EMBEDDING_MODEL_NAME}")

        latest_version = latest_versions[0]

        client.set_registered_model_alias(
            name=EMBEDDING_MODEL_NAME,
            alias="Production",
            version=latest_version.version
        )
        logger.info(f"Model Version {latest_version.version} set to 'Production'.")
        logger.info("--- Embedding Model Fine-Tuning Complete ---")

    except Exception as e:
        logger.error(f"FATAL ERROR during fine-tuning: {e}", exc_info=True)
        raise

if __name__ == "__main__":
    run_embedding_finetune()