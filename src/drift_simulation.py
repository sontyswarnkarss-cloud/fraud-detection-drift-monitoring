"""
Simulation utilities for injecting covariate and concept drift into streaming data slices.
"""

import numpy as np
import pandas as pd
from src.drift_monitor import DriftMonitor
from src.models import train_supervised_model
from src.evaluation import evaluate_predictions

def inject_artificial_drift(
    df, 
    drift_features=('V14', 'V12', 'V10', 'scaled_amount'), 
    amount_multiplier=2.5, 
    pca_shift=1.8,
    random_state=42
):
    """
    Injects realistic concept and covariate drift into a transaction slice.
    
    Realistic Fintech Scenario:
    - Seasonal holiday spending surges average transaction amounts.
    - Organized fraud rings mutate their attack patterns to bypass legacy rules,
      shifting latent PCA components (e.g. V14, V12, V10) where fraud previously clustered.
    """
    np.random.seed(random_state)
    shifted_df = df.copy()
    
    for feat in drift_features:
        if feat in shifted_df.columns:
            if 'amount' in feat.lower():
                # Covariate shift: scale amounts and add variance
                shifted_df[feat] = shifted_df[feat] * amount_multiplier + np.random.normal(0.5, 0.2, len(shifted_df))
            else:
                # Concept shift: distribution mean shift and variance inflation
                shift_direction = 1.0 if np.random.rand() > 0.5 else -1.0
                noise = np.random.normal(shift_direction * pca_shift, 0.6, len(shifted_df))
                shifted_df[feat] = shifted_df[feat] + noise
                
    return shifted_df

def run_drift_experiment(
    model, 
    baseline_ref, 
    clean_target, 
    monitor=None,
    feature_cols=None
):
    """
    Simulates a production streaming environment:
    1. Evaluates a Clean Production Window (Expects: HEALTHY / Low PSI).
    2. Evaluates an Artificially Drifted Production Window (Expects: RETRAIN_ALERT / High PSI).
    3. Triggers simulated automated retraining loop.
    """
    if monitor is None:
        monitor = DriftMonitor(psi_warning=0.10, psi_critical=0.25)
        
    if feature_cols is None:
        feature_cols = [c for c in baseline_ref.columns if c in clean_target.columns]
        
    # Baseline predictions
    ref_preds = model.predict_proba(baseline_ref[feature_cols])[:, 1]
    
    # 1. Clean production window
    clean_preds = model.predict_proba(clean_target[feature_cols])[:, 1]
    clean_feat_drift = monitor.monitor_features(baseline_ref, clean_target, feature_cols)
    clean_pred_drift = monitor.monitor_prediction_drift(ref_preds, clean_preds)
    clean_alert = monitor.evaluate_retraining_trigger(clean_feat_drift, clean_pred_drift)
    
    # 2. Injected drift window
    drifted_target = inject_artificial_drift(clean_target)
    drifted_preds = model.predict_proba(drifted_target[feature_cols])[:, 1]
    drifted_feat_drift = monitor.monitor_features(baseline_ref, drifted_target, feature_cols)
    drifted_pred_drift = monitor.monitor_prediction_drift(ref_preds, drifted_preds)
    drifted_alert = monitor.evaluate_retraining_trigger(drifted_feat_drift, drifted_pred_drift)
    
    return {
        "clean_window": {
            "alert": clean_alert,
            "prediction_drift": clean_pred_drift,
            "top_drifted_features": clean_feat_drift.head(5).to_dict(orient='records')
        },
        "drifted_window": {
            "alert": drifted_alert,
            "prediction_drift": drifted_pred_drift,
            "top_drifted_features": drifted_feat_drift.head(5).to_dict(orient='records'),
            "drifted_target_data": drifted_target
        }
    }
