"""
Model training and scoring module.
Provides implementations for supervised LightGBM with balanced class weights
and unsupervised Isolation Forest for anomaly scoring.
"""

import numpy as np
from sklearn.ensemble import IsolationForest, RandomForestClassifier
import lightgbm as lgb

def train_supervised_model(
    X_train, 
    y_train, 
    model_type='lightgbm', 
    random_state=42
):
    """
    Trains a supervised classifier with cost-sensitive class imbalance weighting.
    """
    if model_type == 'lightgbm':
        clf = lgb.LGBMClassifier(
            class_weight='balanced',
            n_estimators=100,
            learning_rate=0.08,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=random_state,
            n_jobs=-1,
            verbose=-1
        )
        clf.fit(X_train, y_train)
        
        info = {
            "model_type": "LightGBM",
            "imbalance_handling": "class_weight='balanced'",
            "n_estimators": 100,
            "learning_rate": 0.08
        }
        return clf, info
        
    elif model_type == 'random_forest':
        clf = RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight='balanced',
            random_state=random_state,
            n_jobs=-1
        )
        clf.fit(X_train, y_train)
        info = {
            "model_type": "RandomForest",
            "imbalance_handling": "class_weight='balanced'",
            "n_estimators": 100
        }
        return clf, info
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

def train_unsupervised_detector(
    X_train, 
    contamination=0.0017, 
    n_estimators=150, 
    random_state=42
):
    """
    Trains an Isolation Forest unsupervised anomaly detector.
    Zero labels are used during fitting.
    """
    iso_forest = IsolationForest(
        n_estimators=n_estimators,
        max_samples=0.8,
        contamination=contamination,
        random_state=random_state,
        n_jobs=-1
    )
    iso_forest.fit(X_train)
    
    info = {
        "model_type": "IsolationForest",
        "contamination": contamination,
        "n_estimators": n_estimators,
        "supervision": "Unsupervised (Zero Labels Used in Training)"
    }
    return iso_forest, info

def get_anomaly_scores(iso_forest, X):
    """
    Computes normalized anomaly probabilities from an Isolation Forest.
    Inverts the decision function so higher score = higher anomaly probability.
    """
    raw_scores = iso_forest.decision_function(X)
    inverted = -raw_scores
    score_min, score_max = inverted.min(), inverted.max()
    if score_max > score_min:
        norm_scores = (inverted - score_min) / (score_max - score_min)
    else:
        norm_scores = np.zeros_like(inverted)
    return norm_scores
