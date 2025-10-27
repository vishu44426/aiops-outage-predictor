# src/outage_predictor/features/feature_pipeline.py

from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import logging

from.build_features import apply_feature_engineering

logger = logging.getLogger('outage_predictor')

class LogFeatureExtractor(BaseEstimator, TransformerMixin):
    """
    Custom scikit-learn transformer to apply feature engineering 
    to raw log data consistently across training and inference. [8]
    """
    def fit(self, X, y=None):
        """Transformer does not need to learn anything during fit."""
        logger.info("LogFeatureExtractor: Fit called (no learning required).")
        return self

    def transform(self, X):
        """Applies feature engineering logic."""
        logger.debug("LogFeatureExtractor: Transform called.")
        
        # Ensure X is a copy and has the correct Timestamp index for processing
        X_copy = X.copy()
        if not isinstance(X_copy.index, pd.DatetimeIndex):
            # Attempt to set Timestamp as index if it exists as a column (common in MLflow serving)
            if 'Timestamp' in X_copy.columns:
                 X_copy = X_copy.set_index('Timestamp')
                 X_copy.index = pd.to_datetime(X_copy.index)
            else:
                logger.error("Input DataFrame must have a 'Timestamp' index or column for feature engineering.")
                raise ValueError("Input data must contain time information for feature extraction.")

        return apply_feature_engineering(X_copy)