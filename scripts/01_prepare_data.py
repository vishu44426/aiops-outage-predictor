# scripts/01_prepare_data.py

import os
import sys

# Setup path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from outage_predictor.data.synthetic_generator import generate_synthetic_logs, ingest_to_sqlite
from outage_predictor.utils import config

def run_data_preparation():
    """Generates the synthetic dataset and ingests it into SQLite."""
    print("--- Starting Data Generation ---")
    C = config['data']
    
    log_data_df = generate_synthetic_logs()
    ingest_to_sqlite(log_data_df)
    
    # Verification
    outage_count = log_data_df['Outage_Flag'].sum()
    print(f"Total Samples Generated: {len(log_data_df)}")
    print(f"Total Outage Samples (Flag=1) generated: {outage_count} (Approx {outage_count/len(log_data_df)*100:.2f}%)")
    print(f"Data saved to: {C}")
    print("--- Data Preparation Complete ---")

if __name__ == '__main__':
    run_data_preparation()