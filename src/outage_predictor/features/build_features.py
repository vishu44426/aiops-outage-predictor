# src/outage_predictor/features/build_features.py

import pandas as pd
import numpy as np
import logging

from..utils import config

logger = logging.getLogger('outage_predictor')

def apply_feature_engineering(df_raw):
    """
    Applies comprehensive feature engineering including time since last event,
    lag features, interactions, cyclical time, and rolling stats.
    """
    logger.info("Applying COMPREHENSIVE feature engineering...")
    C = config['data']
    ROLLING_WINDOW_MINUTES = C.get('ROLLING_WINDOW_MINUTES', 30)
    WINDOW = f"{ROLLING_WINDOW_MINUTES}min"
    RATE_WINDOW_MINUTES = 5
    RATE_WINDOW = f"{RATE_WINDOW_MINUTES}min"
    # Define lag periods
    LAG_MINUTES = [5, 10]

    if not isinstance(df_raw.index, pd.DatetimeIndex):
        try:
            df_raw.index = pd.to_datetime(df_raw.index)
        except Exception:
            raise ValueError("Feature engineering requires a DatetimeIndex.")

    df_raw = df_raw.copy()

    # --- 1. One-hot encode Severity and Component ---
    SEVERITIES = C.get('SEVERITIES', ['INFO', 'WARNING', 'ERROR', 'CRITICAL'])
    COMPONENTS = C.get('FAILURE_COMPONENTS', ['Database', 'Network', 'API_Gateway', 'Storage', 'CPU'])

    if 'Severity' in df_raw.columns:
        df_raw['Severity'] = pd.Categorical(df_raw['Severity'], categories=SEVERITIES)
    if 'Component' in df_raw.columns:
         df_raw['Component'] = pd.Categorical(df_raw['Component'], categories=COMPONENTS)

    severity_dummies = pd.get_dummies(df_raw['Severity'], prefix='Sev')
    component_dummies = pd.get_dummies(df_raw['Component'], prefix='Comp')

    df_with_dummies = pd.concat([df_raw, severity_dummies, component_dummies], axis=1)

    required_sev_cols = [f'Sev_{s}' for s in SEVERITIES]
    required_comp_cols = [f'Comp_{c}' for c in COMPONENTS]

    for col in required_sev_cols + required_comp_cols:
        if col not in df_with_dummies.columns:
            df_with_dummies[col] = 0
            logger.warning(f"Feature column {col} was missing; added with zeros.")

    # --- Pre-calculate some base series for efficiency ---
    is_error = df_with_dummies['Sev_ERROR']
    is_warning = df_with_dummies['Sev_WARNING']
    is_critical = df_with_dummies['Sev_CRITICAL']
    current_time_series = df_with_dummies.index.to_series()


    # --- 2. Basic Rolling Window Statistics (Severities) ---
    rolling_sev_counts = df_with_dummies[required_sev_cols].rolling(
        window=WINDOW, closed='left', on=df_with_dummies.index
    ).sum().rename(columns=lambda x: f"Rolling_{x.replace('Sev_', '')}_Count")

    rolling_error_std = is_error.rolling(
        window=WINDOW, closed='left', on=df_with_dummies.index
    ).std().fillna(0).rename("Rolling_ERROR_StdDev")

    # --- 3. Exponentially Weighted Moving Average (EWMA) ---
    ewma_error = is_error.ewm(
        span=ROLLING_WINDOW_MINUTES, adjust=False
    ).mean().fillna(0).rename("EWMA_ERROR")
    ewma_error = ewma_error.shift(1, fill_value=0)

    # --- 4. Rate of Change ---
    rolling_error_count_short = is_error.rolling(
        window=RATE_WINDOW, closed='left', on=df_with_dummies.index
    ).sum()
    avg_time_delta = current_time_series.diff().median()
    diff_periods = 1
    if pd.notna(avg_time_delta) and avg_time_delta.total_seconds() > 0:
        diff_periods = max(1, int(pd.Timedelta(minutes=RATE_WINDOW_MINUTES) / avg_time_delta))
    rate_of_change_error = rolling_error_count_short.diff(
        periods=diff_periods
        ).fillna(0).rename("Rate_Change_ERROR_Count")

    # --- 5. Component-Specific Rolling Counts ---
    component_features = []
    for comp_col in required_comp_cols:
        comp_name = comp_col.replace('Comp_', '')
        is_component = df_with_dummies[comp_col]

        comp_error_interaction = is_component * is_error
        rolling_comp_error = comp_error_interaction.rolling(
            window=WINDOW, closed='left', on=df_with_dummies.index
        ).sum().fillna(0).rename(f"Rolling_ERROR_{comp_name}_Count")
        component_features.append(rolling_comp_error)

        comp_warn_interaction = is_component * is_warning
        rolling_comp_warn = comp_warn_interaction.rolling(
            window=WINDOW, closed='left', on=df_with_dummies.index
        ).sum().fillna(0).rename(f"Rolling_WARNING_{comp_name}_Count")
        component_features.append(rolling_comp_warn)
    df_component_features = pd.concat(component_features, axis=1)

    # --- 6. Component-Severity Interaction Rolling Counts ---
    interaction_features = []
    for comp_col in required_comp_cols:
        comp_name = comp_col.replace('Comp_', '')
        is_component = df_with_dummies[comp_col]
        for sev_col in ['Sev_WARNING', 'Sev_ERROR', 'Sev_CRITICAL']:
            sev_name = sev_col.replace('Sev_', '')
            is_severity = df_with_dummies[sev_col]
            interaction_col = is_component * is_severity
            rolling_interaction = interaction_col.rolling(
                window=WINDOW, closed='left', on=df_with_dummies.index
            ).sum().fillna(0).rename(f"Rolling_{sev_name}_{comp_name}_Interaction_Count")
            interaction_features.append(rolling_interaction)
    df_interaction_features = pd.concat(interaction_features, axis=1)

    # --- 7. [NEW] Time Since Last Event Features ---
    time_since_features = []
    logger.info("Generating time since last event features...")

    # Calculate time since the last overall CRITICAL event
    last_critical_time = current_time_series.where(is_critical == 1).ffill()
    time_since_critical = (current_time_series - last_critical_time).dt.total_seconds().fillna(-1).rename("Time_Since_Last_CRITICAL_Sec") # Use -1 for no prior event
    time_since_features.append(time_since_critical)

    # Calculate time since the last ERROR for each component
    for comp_col in required_comp_cols:
        comp_name = comp_col.replace('Comp_', '')
        is_component = df_with_dummies[comp_col]
        comp_error_interaction = is_component * is_error
        last_comp_error_time = current_time_series.where(comp_error_interaction == 1).ffill()
        time_since_comp_error = (current_time_series - last_comp_error_time).dt.total_seconds().fillna(-1).rename(f"Time_Since_Last_ERROR_{comp_name}_Sec")
        time_since_features.append(time_since_comp_error)

    df_time_since_features = pd.concat(time_since_features, axis=1)


    # --- 8. [NEW] Lag Features ---
    lagged_features = []
    logger.info("Generating lag features...")
    # Add lags for key rolling counts and EWMA
    features_to_lag = {
        'Rolling_ERROR_Count': rolling_sev_counts['Rolling_ERROR_Count'],
        'Rolling_CRITICAL_Count': rolling_sev_counts['Rolling_CRITICAL_Count'],
        'EWMA_ERROR': ewma_error,
        'Rate_Change_ERROR_Count': rate_of_change_error
    }
    # Add component error counts to lag features
    for col in df_component_features.columns:
        if 'ERROR' in col:
             features_to_lag[col] = df_component_features[col]

    # Calculate lags based on approximate number of steps
    if pd.notna(avg_time_delta) and avg_time_delta.total_seconds() > 0:
        for lag_min in LAG_MINUTES:
            lag_periods = max(1, int(pd.Timedelta(minutes=lag_min) / avg_time_delta))
            for name, series in features_to_lag.items():
                lagged_series = series.shift(lag_periods).fillna(0).rename(f"{name}_Lag_{lag_min}min")
                lagged_features.append(lagged_series)
    else:
        logger.warning("Could not calculate median time delta; skipping lag features.")

    df_lagged_features = pd.concat(lagged_features, axis=1)


    # --- 9. Cyclical Time features ---
    time_features = pd.DataFrame(index=df_raw.index)
    current_hour = df_raw.index.hour
    current_dow = df_raw.index.dayofweek
    time_features['Hour_sin'] = np.sin(2 * np.pi * current_hour/24.0)
    time_features['Hour_cos'] = np.cos(2 * np.pi * current_hour/24.0)
    time_features['DayOfWeek_sin'] = np.sin(2 * np.pi * current_dow/7.0)
    time_features['DayOfWeek_cos'] = np.cos(2 * np.pi * current_dow/7.0)


    # --- 10. Final Feature Set Combination ---
    X_features = pd.concat([
        time_features,            # Cyclical time
        rolling_sev_counts,       # Basic severity counts
        rolling_error_std,        # Error volatility
        ewma_error,               # Weighted recent errors
        rate_of_change_error,     # Error acceleration
        df_component_features,    # Component error/warn counts
        df_interaction_features,  # Component-Severity interaction counts
        df_time_since_features,   # Time since last critical events
        df_lagged_features        # Lagged features
    ], axis=1)

    # Fill initial NaNs introduced by rolling, shifts, diffs etc.
    X_features = X_features.fillna(0)
    # Replace any potential infinite values (e.g., from std dev on constant)
    X_features.replace([np.inf, -np.inf], 0, inplace=True)


    logger.info(f"Generated {len(X_features.columns)} total features. Example: {list(X_features.columns[:5])}...")
    X_features = X_features.select_dtypes(include=np.number)

    return X_features