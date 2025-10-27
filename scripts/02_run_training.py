# scripts/02_run_training.py

import os
import sys

# Setup path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from outage_predictor.models.train import train_model

if __name__ == '__main__':
    print("--- Starting Full Training Pipeline ---")
    print("NOTE: Ensure MLflow Tracking Server is running at http://127.0.0.1:5000")
    try:
        train_model()
    except Exception as e:
        print(f"\nTraining failed. Check MLflow server or data preparation: {e}")