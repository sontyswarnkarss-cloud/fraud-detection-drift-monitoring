"""
Data loading and preprocessing utilities for credit card transaction data.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "creditcard.csv")

def load_credit_card_data(filepath=DEFAULT_DATA_PATH):
    """
    Loads creditcard.csv and performs initial validation checks.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Dataset not found at {filepath}. Run `python download_data.py` first."
        )
    
    df = pd.read_csv(filepath)
    expected_cols = {'Time', 'Amount', 'Class'}.union({f'V{i}' for i in range(1, 29)})
    if not expected_cols.issubset(set(df.columns)):
        missing = expected_cols - set(df.columns)
        raise ValueError(f"Dataset missing expected columns: {missing}")
        
    return df

def get_imbalance_statistics(df):
    """
    Computes class counts and exact imbalance proportions.
    """
    total = len(df)
    counts = df['Class'].value_counts()
    legit_count = counts.get(0, 0)
    fraud_count = counts.get(1, 0)
    
    stats = {
        "total_transactions": int(total),
        "legitimate_count": int(legit_count),
        "fraud_count": int(fraud_count),
        "fraud_percentage": float((fraud_count / total) * 100),
        "imbalance_ratio": float(legit_count / max(1, fraud_count))
    }
    return stats

def preprocess_and_split(
    df, 
    test_size=0.25, 
    random_state=42, 
    split_strategy='stratified',
    time_cutoff_ratio=0.70
):
    """
    Transforms data with:
    1. Cyclical time feature: 'hour_of_day' = (Time // 3600) % 24
    2. Outlier-resistant scaling: RobustScaler for 'Amount'
    3. Stratified or temporal window splitting.
    """
    data = df.copy()
    
    # Feature Engineering: Convert monotonic Time (0-172800s) to 24-hr daily cyclical periodicity
    data['hour_of_day'] = (data['Time'] // 3600) % 24
    
    # Outlier-resistant scaling for transaction Amount
    scaler = RobustScaler()
    data['scaled_amount'] = scaler.fit_transform(data[['Amount']])
    
    # Feature columns: 28 PCA features + engineered amount & cyclical hour
    feature_cols = [f'V{i}' for i in range(1, 29)] + ['scaled_amount', 'hour_of_day']
    X = data[feature_cols]
    y = data['Class']
    
    if split_strategy == 'temporal':
        # Sort by chronological arrival order
        sorted_idx = data['Time'].sort_values().index
        data_sorted = data.loc[sorted_idx].reset_index(drop=True)
        
        cutoff_idx = int(len(data_sorted) * time_cutoff_ratio)
        train_df = data_sorted.iloc[:cutoff_idx]
        val_df = data_sorted.iloc[cutoff_idx:]
        
        X_train = train_df[feature_cols]
        y_train = train_df['Class']
        X_val = val_df[feature_cols]
        y_val = val_df['Class']
        
        reference_window = X_train.copy()
        later_window = X_val.copy()
    else:
        # Stratified train-test split for balanced comparative evaluation
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        reference_window = X_train.copy()
        later_window = X_val.copy()
        
    return {
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "reference_window": reference_window,
        "later_window": later_window,
        "scaler": scaler,
        "feature_names": feature_cols
    }
