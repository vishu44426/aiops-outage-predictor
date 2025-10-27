# src/outage_predictor/data/synthetic_generator.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from faker import Faker
from sqlalchemy import create_engine
import os
import logging

from..utils import config

logger = logging.getLogger('outage_predictor')
fake = Faker()

# Helper function for realistic log messages
def _generate_log_message(component, severity):
    """Generates a realistic fake log message based on the component."""
    
    # Use a simple severity-based message for the final CRITICAL outage log
    if severity == 'CRITICAL':
        return f"CRITICAL FAILURE: {component} System Down or Unresponsive"
        
    try:
        if component == 'Network':
            action = fake.random_element(elements=(
                ('Connection timed out', 'ERROR'), 
                ('Packet dropped', 'ERROR'), 
                ('Successful handshake', 'INFO'), 
                ('Firewall block', 'WARNING'), 
                ('High latency detected', 'WARNING'), 
                ('Connection established', 'INFO')
            ))
            protocol = fake.random_element(elements=('TCP', 'UDP', 'ICMP'))
            src_ip = fake.ipv4()
            dest_ip = fake.ipv4()
            port = fake.random_int(min=1024, max=65535)
            # Return a message that matches the randomly chosen severity
            if severity in ['ERROR', 'WARNING'] and action[1] in ['ERROR', 'WARNING']:
                return f"{action[0]}: {protocol} from {src_ip} to {dest_ip} on port {port}"
            else:
                return f"INFO: {protocol} packet from {src_ip} to {dest_ip}. Status: OK"

        elif component == 'Database':
            action = fake.random_element(elements=(
                ('Query executed', 'INFO'), 
                ('Slow query detected', 'WARNING'), 
                ('Deadlock detected', 'ERROR'), 
                ('Connection pool exhausted', 'CRITICAL')
            ))
            table = fake.random_element(elements=('users', 'orders', 'products', 'logs'))
            if severity == 'INFO' and action[1] == 'INFO':
                 return f"{action[0]} on table '{table}'. Rows: {fake.random_int(min=0, max=5000)}"
            else:
                return f"{severity}: {action[0]} on table '{table}'. Query time: {fake.random_int(min=100, max=5000)}ms"

        elif component == 'CPU':
            if severity == 'INFO':
                return f"CPU load normal at {fake.random_int(min=10, max=40)}%"
            else:
                return f"CPU load {severity}: threshold exceeded {fake.random_int(min=75, max=99)}%"
        
        elif component == 'Storage':
            if severity == 'INFO':
                return f"Disk usage OK at {fake.random_int(min=20, max=50)}% on '/var/log'"
            else:
                return f"Disk usage {severity}: {fake.random_int(min=85, max=99)}% on '/var/log'"
        
        # Default for API_Gateway or other components
        else:
            return fake.sentence(nb_words=5)
            
    except Exception:
        # Fallback in case of any Faker error
        return fake.sentence(nb_words=5)


def generate_synthetic_logs():
    """
    Generates synthetic server log data with injected outage precursor patterns 
    based on the Rules Engine approach.
    """
    logger.info("Starting synthetic log generation...")
    C = config['data']
    
    # Load parameters with defaults for robustness
    TOTAL_SAMPLES = C.get('TOTAL_SAMPLES', 10000)
    START_TIME = datetime.strptime(C.get('START_TIME', "2024-01-01 00:00:00"), "%Y-%m-%d %H:%M:%S")
    LOG_RATE_PER_MIN = C.get('LOG_RATE_PER_MIN', 1.0)
    FAILURE_COMPONENTS = C.get('FAILURE_COMPONENTS', ['database', 'network'])
    SEVERITIES = C.get('SEVERITIES', ['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    SEVERITY_PROBS = C.get('SEVERITY_PROBS', [0.90, 0.07, 0.025, 0.005])
    NUM_OUTAGES = C.get('NUM_OUTAGES', 5)
    PRECURSOR_WINDOW_MINUTES = C.get('PRECURSOR_WINDOW_MINUTES', 15)
    PRECURSOR_ERROR_RATE = C.get('PRECURSOR_ERROR_RATE', 0.7)

    # Calculate end time based on total samples and log rate
    total_duration_minutes = TOTAL_SAMPLES / LOG_RATE_PER_MIN
    END_TIME = START_TIME + timedelta(minutes=total_duration_minutes)

    data = []
    current_time = START_TIME

    # Step 1: Generate background noise (Normal Operations)
    time_increment = timedelta(seconds=60 / LOG_RATE_PER_MIN)
    while current_time < END_TIME:
        # Generate severity based on normal distribution bias
        severity = np.random.choice(SEVERITIES, p=SEVERITY_PROBS)
        component = np.random.choice(FAILURE_COMPONENTS)
        
        # Call new function for realistic message
        message = _generate_log_message(component, severity)
        
        data.append({
            'Timestamp': current_time, 
            'Severity': severity, 
            'Component': component, 
            'Message': message,
            'Outage_Flag': 0 
        })
        current_time += time_increment

    df = pd.DataFrame(data).set_index('Timestamp').sort_index()
    logger.info(f"Generated {len(df)} initial log entries.")

    # Step 2: Inject precursor patterns (Outage Flag = 1)
    
    precursor_probs = [
        (1.0 - PRECURSOR_ERROR_RATE) / 2, # INFO
        (1.0 - PRECURSOR_ERROR_RATE) / 2, # WARNING
        PRECURSOR_ERROR_RATE / 2,         # ERROR
        PRECURSOR_ERROR_RATE / 2          # CRITICAL
    ]
    
    for i in range(NUM_OUTAGES):
        # Choose a random time index for an outage
        outage_index = np.random.randint(int(len(df) * 0.1), int(len(df) * 0.9))
        outage_time = df.index[outage_index] 
        outage_component = np.random.choice(FAILURE_COMPONENTS)
        
        precursor_start = outage_time - timedelta(minutes=PRECURSOR_WINDOW_MINUTES)
        
        precursor_mask = (df.index >= precursor_start) & (df.index <= outage_time)
        df.loc[precursor_mask, 'Outage_Flag'] = 1
        
        component_mask = (df['Component'] == outage_component)
        full_mask = precursor_mask & component_mask
        
        num_precursor_logs = full_mask.sum()

        if num_precursor_logs > 0:
            # Inject new severities
            new_severities = np.random.choice(
                SEVERITIES, 
                size=num_precursor_logs,
                p=precursor_probs
            )
            df.loc[full_mask, 'Severity'] = new_severities
            
            # Generate new realistic messages for the new severities
            new_messages = [
                _generate_log_message(outage_component, sev) for sev in new_severities
            ]
            df.loc[full_mask, 'Message'] = new_messages

        # 2c. Set the final outage log entry (T)
        df.loc[outage_time, 'Severity'] = 'CRITICAL'
        df.loc[outage_time, 'Component'] = outage_component
        df.loc[outage_time, 'Message'] = f"CRITICAL FAILURE: {outage_component} System Down"
        
        logger.info(f"Injected Outage {i+1}/{NUM_OUTAGES} for {outage_component} at {outage_time}")

    return df.reset_index()

def ingest_to_sqlite(df):
    """Saves the generated DataFrame to a SQLite database."""
    C = config['data']
    db_path = C.get('DB_PATH', "data/log_outage_data.db")
    table_name = C.get('TABLE_NAME', "server_logs")

    # Ensure the data directory exists
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    
    try:
        # Create the database engine
        engine = create_engine(f"sqlite:///{db_path}")

        # Write the DataFrame to SQLite
        df.to_sql(table_name, engine, if_exists='replace', index=False)
        logger.info(f"Data successfully ingested to SQLite: {db_path} in table '{table_name}'.")
    except Exception as e:
        logger.error(f"Failed to ingest data to SQLite: {e}")
        raise