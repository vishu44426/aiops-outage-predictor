# scripts/03_simulate_inference.py

import os
import sys
from datetime import datetime, timedelta

# Setup path for local imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from outage_predictor.deployment.predictor_service import predict_realtime
from outage_predictor.deployment.mlflow_utils import transition_model_to_production
from outage_predictor.utils import config

def simulate_realtime_request():
    """
    Simulates a production request by creating a raw log input window 
    that might trigger an alert.
    """
    
    target_component = "CPU"
    current_time = datetime.now()
    
    # Create a batch of recent log entries (e.g., 5 logs over 5 minutes)
    # This simulates a spike in ERROR logs for the 'CPU' component
    recent_logs = [
        {
            "Timestamp": (current_time - timedelta(minutes=4)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": "Network",
            "Severity": "INFO",
            "Message": "Packet received"
        },
        {
            "Timestamp": (current_time - timedelta(minutes=3)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": target_component,
            "Severity": "WARNING",
            "Message": "CPU load at 70%"
        },
        {
            "Timestamp": (current_time - timedelta(minutes=2)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": target_component,
            "Severity": "ERROR",
            "Message": "CPU temperature threshold warning"
        },
        {
            "Timestamp": (current_time - timedelta(minutes=1)).strftime("%Y-%m-%d %H:%M:%S"),
            "Component": target_component,
            "Severity": "ERROR",
            "Message": "CPU fan speed low"
        },
        {
            "Timestamp": current_time.strftime("%Y-%m-%d %H:%M:%S"),
            "Component": target_component,
            "Severity": "CRITICAL",
            "Message": "CPU OVERHEAT SHUTDOWN IMMINENT"
        }
    ]
    
    # Prediction uses the model in the MLflow Registry
    result = predict_realtime(recent_logs)
    
    print("\n--- Production Inference Simulation ---")
    if "error" in result:
        print(f"Prediction Error: {result['error']}")
    else:
        print(f"Prediction Time: {result['prediction_timestamp']}")
        print(f"Correlation ID (for monitoring): {result['correlationid']}")
        print(f"Outage Probability: {result['prediction_probability']:.4f}")
        print(f"Alert Status: {result['alert_status']}")
        
def run_deployment_tests():
    """
    Simulates the MLOps process of promoting a model and testing inference.
    """
    print("--- MLOps Deployment Management ---")
    
    # 1. Transition the latest trained model version to Production [14]
    transition_model_to_production() 
    
    # 2. Simulate Real-Time Request (testing the production model)
    simulate_realtime_request()
    
    print("--- Simulation Complete ---")

if __name__ == '__main__':
    run_deployment_tests()