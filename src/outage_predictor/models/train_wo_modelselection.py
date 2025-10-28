# src/outage_predictor/models/train.py

import mlflow
import xgboost as xgb
from sklearn.pipeline import Pipeline, make_pipeline # Import make_pipeline
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

from..utils import config
# Use the data loader that gets features, AND load raw data + split function
from..data.data_loader import get_chronological_splits, load_data_from_db, chronological_split
from..features.feature_pipeline import LogFeatureExtractor

logger = logging.getLogger('outage_predictor')

def calculate_class_weights(y_data):
    """Calculates scale_pos_weight for XGBoost."""
    count_no_outage = sum(y_data == 0)
    count_outage = sum(y_data == 1)
    if count_outage == 0:
        logger.warning("No outage events found. Using 1.0.")
        return 1.0
    scale_pos_weight = count_no_outage / count_outage
    logger.info(f"Class imbalance ratio... Setting scale_pos_weight = {scale_pos_weight:.2f}")
    return scale_pos_weight

def train_model():
    """
    Trains XGBoost using pre-computed features BUT logs a pipeline
    that includes the feature extractor for inference.
    """
    logger.info("--- Starting XGBoost Training (using Feature Store for fit, logging full pipeline) ---")

    C_MLOPS = config['mlops']
    C_MODEL = config['model']
    MODEL_NAME = C_MLOPS.get('MODEL_NAME', "Outage_Predictor_GBM")

    # 1. Load Pre-computed Features AND Raw Data
    logger.info("Loading pre-computed features and splitting data...")
    X_train_feat, X_val_feat, X_test_feat, y_train, y_val, y_test = get_chronological_splits()
    logger.info(f"Loaded features shapes: Train={X_train_feat.shape}, Test={X_test_feat.shape}")

    logger.info("Loading raw data subset for pipeline definition...")
    df_raw_full = load_data_from_db() # Load raw data
    y_raw = df_raw_full['Outage_Flag']
    X_raw = df_raw_full.drop(columns=['Outage_Flag', 'Message']).copy()
    # Apply same chronological split to raw data
    X_train_raw, _, X_test_raw, _, _, _ = chronological_split(X_raw, y_raw, config['data']['TRAIN_RATIO'], config['data']['VAL_RATIO'])

    # 2. Calculate Imbalance Weights
    scale_pos_weight = calculate_class_weights(y_train)

    # 3. Setup MLflow Tracking
    mlflow.set_tracking_uri(C_MLOPS.get('MLFLOW_TRACKING_URI'))
    mlflow.set_experiment("/AIOps_Outage_Prediction") # Ensure experiment name is correct

    mlflow.xgboost.autolog(disable=True) # Disable autologging
    logger.info("MLflow Autologging disabled; manual logging enabled.")

    # 4. Start MLflow Run
    with mlflow.start_run(run_name="XGBoost_FeatureStore_FullPipe") as run:
        logger.info(f"MLflow Run ID: {run.info.run_id}")

        # Configure XGBoost Classifier
        xgb_params = C_MODEL.get('XGBOOST_PARAMS', {})
        xgb_params['scale_pos_weight'] = scale_pos_weight
        classifier = xgb.XGBClassifier(**xgb_params)

        # 5. Define the FULL pipeline (for logging/inference)
        pipeline = make_pipeline(
            LogFeatureExtractor(),
            StandardScaler(),
            classifier
        )
        logger.info("Pipeline constructed: LogFeatureExtractor -> StandardScaler -> XGBoost.")

        # 6. Fit the pipeline steps selectively
        logger.info("Fitting pipeline steps...")
        # Fit Feature Extractor (needs raw data, though it doesn't learn)
        pipeline.named_steps['logfeatureextractor'].fit(X_train_raw, y_train)
        # Fit Scaler and Classifier (needs pre-computed features)
        pipeline.named_steps['standardscaler'].fit(X_train_feat, y_train)
        pipeline.named_steps['xgbclassifier'].fit( # Access classifier by its default name
             pipeline.named_steps['standardscaler'].transform(X_train_feat), # Pass scaled features
             y_train
        )
        logger.info("Pipeline fitting complete.")

        # 7. Evaluation on Held-Out Test Set (using features)
        logger.info("Evaluating model on held-out test set features...")
        # Create a temporary pipeline for evaluation (Scaler + Classifier)
        scaler_and_classifier = Pipeline(steps=[
            ('standardscaler', pipeline.named_steps['standardscaler']),
            ('xgbclassifier', pipeline.named_steps['xgbclassifier'])
        ])
        y_pred = scaler_and_classifier.predict(X_test_feat)
        y_pred_proba = scaler_and_classifier.predict_proba(X_test_feat)[:, 1] # Proba for class 1

        # Calculate metrics
        test_recall = recall_score(y_test, y_pred, pos_label=1, zero_division=0)
        test_precision = precision_score(y_test, y_pred, pos_label=1, zero_division=0)
        test_f1 = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
        test_auc = 0.0
        if len(np.unique(y_test)) > 1:
            test_auc = roc_auc_score(y_test, y_pred_proba)
        else:
            logger.warning("Only one class present in y_test. Skipping ROC AUC calculation.")

        # Log metrics manually
        mlflow.log_metric("minority_class_recall", test_recall)
        mlflow.log_metric("minority_class_precision", test_precision)
        mlflow.log_metric("f1_score", test_f1)
        mlflow.log_metric("roc_auc", test_auc)
        mlflow.log_param("imbalance_scale_pos_weight", scale_pos_weight)
        mlflow.log_params(xgb_params)

        # Log Confusion Matrix (optional)
        # ... (code to plot and log confusion matrix remains the same)
        try:
            cm = confusion_matrix(y_test, y_pred, labels=pipeline.classes_)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=pipeline.classes_)
            disp.plot(); plt.title("Test Set Confusion Matrix"); cm_path = "test_confusion_matrix.png"; plt.savefig(cm_path)
            mlflow.log_artifact(cm_path)
            logger.info("Confusion matrix saved and logged.")
        except Exception as e: logger.warning(f"Could not save confusion matrix plot: {e}")


        logger.info("-" * 50)
        logger.info(f"Model Training Complete (Run ID: {run.info.run_id})")
        logger.info(f"  Recall: {test_recall:.4f}, Precision: {test_precision:.4f}, F1: {test_f1:.4f}, AUC: {test_auc:.4f}")
        logger.info("-" * 50)

        # 8. Model Persistence and Registration
        logger.info("Logging and registering the full pipeline...")
        try:
            # Infer signature using RAW data and the FULL pipeline's predict_proba
            X_train_raw_serving = X_train_raw.reset_index().head(10)
            # Use predict_proba for signature output
            y_pred_proba_sig = pipeline.predict_proba(X_train_raw.head(10))

            signature = infer_signature(X_train_raw_serving, y_pred_proba_sig)
            input_example = X_train_raw_serving.head(1)

            # Log the full pipeline object
            mlflow.sklearn.log_model(
                sk_model=pipeline, # Log the full pipeline
                artifact_path="outage_predictor_pipeline_full", # Use consistent path
                registered_model_name=MODEL_NAME,
                signature=signature,
                input_example=input_example
            )
            logger.info(f"Successfully registered winning FULL pipeline 'XGBoost' as '{MODEL_NAME}'.")

        except Exception as e:
            logger.error(f"Failed to register the model: {e}", exc_info=True)