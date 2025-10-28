# src/outage_predictor/models/train.py

import mlflow
import xgboost as xgb

from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    recall_score, precision_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)
from mlflow.models.signature import infer_signature
import pandas as pd
import logging
import matplotlib.pyplot as plt
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from..utils import config
# Use the data loader that gets features, BUT we also need raw data for signature/example
from..data.data_loader import get_chronological_splits, load_data_from_db, chronological_split
from..features.feature_pipeline import LogFeatureExtractor

logger = logging.getLogger('outage_predictor')

# ... (calculate_class_weights, get_candidate_models remain the same) ...
def calculate_class_weights(y_data):
    count_no_outage = sum(y_data == 0)
    count_outage = sum(y_data == 1)
    if count_outage == 0: return 1.0
    scale_pos_weight = count_no_outage / count_outage
    logger.info(f"Class imbalance ratio (0:1)... Setting scale_pos_weight = {scale_pos_weight:.2f}")
    return scale_pos_weight

def get_candidate_models(scale_pos_weight):
    C_MODEL = config['model']
    xgb_params = C_MODEL.get('XGBOOST_PARAMS', {})
    xgb_params['scale_pos_weight'] = scale_pos_weight
    return {
        "LogisticRegression": LogisticRegression(class_weight='balanced', random_state=42, solver='liblinear'),
        "RandomForest": RandomForestClassifier(class_weight='balanced', random_state=42, n_estimators=100),
        "XGBoost": xgb.XGBClassifier(**xgb_params)
    }

def train_model():
    """
    Trains models using pre-computed features BUT logs a pipeline
    that includes the feature extractor for inference.
    """
    logger.info("--- Starting Model Training (using Feature Store for fit, logging full pipeline) ---")

    C_MLOPS = config['mlops']
    MODEL_NAME = C_MLOPS.get('MODEL_NAME', "Outage_Predictor_GBM")

    # 1. Load Pre-computed Features AND Raw Data
    logger.info("Loading pre-computed features and splitting data...")
    X_train_feat, X_val_feat, X_test_feat, y_train, y_val, y_test = get_chronological_splits()
    logger.info(f"Loaded features shapes: Train={X_train_feat.shape}, Test={X_test_feat.shape}")

    # Load corresponding RAW data needed for signature/example and fitting the extractor
    logger.info("Loading raw data subset for pipeline definition...")
    df_raw_full = load_data_from_db() # Assumes load_data_from_db returns the full raw df
    y_raw = df_raw_full['Outage_Flag']
    X_raw = df_raw_full.drop(columns=['Outage_Flag', 'Message']).copy()
    # Apply same chronological split to raw data
    X_train_raw, _, X_test_raw, _, _, _ = chronological_split(X_raw, y_raw, config['data']['TRAIN_RATIO'], config['data']['VAL_RATIO'])


    # 2. Calculate Imbalance Weights
    scale_pos_weight = calculate_class_weights(y_train)

    # 3. Get Candidate Models
    candidate_models = get_candidate_models(scale_pos_weight)

    # 4. Setup MLflow
    mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("/AIOps_Outage_Prediction")

    all_model_results = []
    primary_metric = "f1_score"
    best_model_pipeline = None # This will store the pipeline OBJECT to be logged
    best_model_name = ""
    best_score = -1.0
    best_run_id = ""

    # 5. Start parent run
    with mlflow.start_run(run_name="Model_Selection_FeatureStore_FullPipe") as parent_run:
        logger.info(f"Parent MLflow Run ID: {parent_run.info.run_id}")

        for model_name, classifier in candidate_models.items():
            with mlflow.start_run(run_name=model_name, nested=True) as child_run:
                logger.info(f"--- Training Model: {model_name} (Run ID: {child_run.info.run_id}) ---")

                # Define the FULL pipeline including the feature extractor
                pipeline = make_pipeline(
                    LogFeatureExtractor(), # Re-added for inference compatibility
                    StandardScaler(),
                    classifier
                )
                logger.info("Pipeline constructed: LogFeatureExtractor -> StandardScaler -> Classifier.")

                # Train the pipeline steps *using the appropriate data*
                logger.info("Fitting pipeline steps...")
                # Fit Feature Extractor (needs raw data, though it doesn't learn)
                pipeline.named_steps['logfeatureextractor'].fit(X_train_raw, y_train)
                # Fit Scaler and Classifier (needs pre-computed features)
                # We access steps by name and fit them individually on the correct data
                pipeline.named_steps['standardscaler'].fit(X_train_feat, y_train)
                pipeline.named_steps[classifier.__class__.__name__.lower()].fit( # Get classifier name dynamically
                     pipeline.named_steps['standardscaler'].transform(X_train_feat), # Pass scaled features
                     y_train
                )
                logger.info("Pipeline fitting complete.")


                # Evaluate using the test set features
                logger.info("Evaluating model on held-out test set features...")
                # For evaluation, we only need scaler + classifier part
                scaler_and_classifier = Pipeline(steps=[
                    ('standardscaler', pipeline.named_steps['standardscaler']),
                    (classifier.__class__.__name__.lower(), pipeline.named_steps[classifier.__class__.__name__.lower()])
                ])
                y_pred = scaler_and_classifier.predict(X_test_feat)
                y_pred_proba = scaler_and_classifier.predict_proba(X_test_feat)[:, 1]

                # ... (Calculate and log metrics - unchanged)
                test_recall=recall_score(y_test,y_pred,pos_label=1,zero_division=0); test_precision=precision_score(y_test,y_pred,pos_label=1,zero_division=0); test_f1=f1_score(y_test,y_pred,pos_label=1,zero_division=0); test_auc=0.0
                if len(np.unique(y_test)) > 1: test_auc=roc_auc_score(y_test,y_pred_proba)
                mlflow.log_metric("minority_class_recall",test_recall); mlflow.log_metric("minority_class_precision",test_precision); mlflow.log_metric("f1_score",test_f1); mlflow.log_metric("roc_auc",test_auc); mlflow.log_param("model_type",model_name)
                all_model_results.append({"model_name":model_name, "run_id":child_run.info.run_id, "f1_score":test_f1, "roc_auc":test_auc, "recall":test_recall, "precision":test_precision})

                # Check best model
                current_score = test_f1 if primary_metric == "f1_score" else test_auc
                if current_score > best_score:
                    best_score = current_score
                    best_model_pipeline = pipeline
                    best_model_name = model_name
                    best_run_id = child_run.info.run_id


    # 6. Print comparison and register the winning *full* pipeline
    if best_model_pipeline is None: logger.error("No models trained."); return
    logger.info("--- Model Selection Complete ---")
    # ... (Print comparison table - unchanged)
    logger.info(f"\nWinning Model: {best_model_name} (Run ID: {best_run_id})")

    try:
        # Infer signature using RAW data and the FULL pipeline's predict_proba
        X_train_raw_serving = X_train_raw.reset_index().head(10) # Format for signature
        y_pred_proba_sig = best_model_pipeline.predict_proba(X_train_raw.head(10)) # Use full pipeline

        signature = infer_signature(X_train_raw_serving, y_pred_proba_sig)
        input_example = X_train_raw_serving.head(1)

        # Log the full pipeline (Extractor -> Scaler -> Classifier)
        mlflow.sklearn.log_model(
            sk_model=best_model_pipeline, # Log the pipeline containing the extractor
            artifact_path="outage_predictor_pipeline_full", # New path name
            registered_model_name=MODEL_NAME,
            signature=signature, # Signature based on raw input, probability output
            input_example=input_example # Example based on raw input
        )
        logger.info(f"Successfully registered winning FULL pipeline '{best_model_name}' as '{MODEL_NAME}'.")

    except Exception as e:
        logger.error(f"Failed to register the winning model: {e}", exc_info=True)