"""
Evaluation and Business Cost-Optimization Module
------------------------------------------------
Provides:
1. Precision-Recall AUC (PR-AUC) and ROC-AUC metrics.
2. Rigorous explanation of PR-AUC vs ROC-AUC under extreme imbalance.
3. Business Cost Matrix Analysis (False Positive Cost vs False Negative Cost).
4. Threshold optimization to minimize financial losses.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import (
    precision_recall_curve,
    roc_curve,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

def evaluate_predictions(y_true, y_scores, threshold=0.5):
    """
    Computes comprehensive evaluation metrics for imbalanced fraud classification.
    
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary labels (0 or 1).
    y_scores : array-like
        Predicted fraud probability or normalized anomaly score in [0, 1].
    threshold : float
        Decision threshold for binary classification.
        
    Returns:
    --------
    dict with metrics, curves, and confusion matrix.
    """
    y_pred = (y_scores >= threshold).astype(int)
    
    pr_auc = float(average_precision_score(y_true, y_scores))
    roc_auc = float(roc_auc_score(y_true, y_scores))
    
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    
    prec_curve, rec_curve, pr_thresholds = precision_recall_curve(y_true, y_scores)
    fpr_curve, tpr_curve, roc_thresholds = roc_curve(y_true, y_scores)
    
    return {
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "specificity": specificity,
        "confusion_matrix": {
            "TN": int(tn),
            "FP": int(fp),
            "FN": int(fn),
            "TP": int(tp)
        },
        "curves": {
            "precision_curve": prec_curve,
            "recall_curve": rec_curve,
            "fpr_curve": fpr_curve,
            "tpr_curve": tpr_curve
        }
    }

def explain_imbalance_metrics():
    """
    Returns an academic and practical justification of why PR-AUC is superior
    to ROC-AUC on extremely imbalanced fraud datasets.
    """
    explanation = (
        "Why PR-AUC (Average Precision) is essential over ROC-AUC:\n"
        "---------------------------------------------------------\n"
        "1. In Credit Card Fraud detection, the positive class (fraud) is extremely rare (~0.17%).\n"
        "2. ROC curves plot True Positive Rate (TPR = TP / P) against False Positive Rate (FPR = FP / N).\n"
        "   Because the number of legitimate transactions N is overwhelmingly large (e.g. ~284,000),\n"
        "   even if the model produces 500 False Positives, FPR = 500 / 284,000 ≈ 0.00176.\n"
        "   This negligible FPR shifts the ROC curve tightly against the top-left corner, yielding an\n"
        "   artificially optimistic ROC-AUC of 0.98+ while hundreds of customers suffer false declines.\n"
        "3. Precision-Recall curves plot Precision (TP / (TP + FP)) against Recall (TP / (TP + FN)).\n"
        "   The PR curve completely excludes the massive True Negative count (TN). Every false positive\n"
        "   directly degrades Precision (e.g. 100 TP / (100 TP + 500 FP) = 16.7% Precision).\n"
        "4. Conclusion: PR-AUC reflects real-world operational utility by penalizing false alarms,\n"
        "   making it the definitive benchmark for imbalanced fraud systems."
    )
    return explanation

def optimize_cost_threshold(y_true, y_scores, cost_fp=10.0, cost_fn=150.0, num_thresholds=200):
    """
    Conducts cost-sensitive business optimization to identify the optimal
    decision threshold that minimizes financial and operational loss.
    
    Cost Function:
        Total Cost = (FP * Cost_FP) + (FN * Cost_FN)
        
    Parameters:
    -----------
    y_true : array-like
        Ground truth binary labels.
    y_scores : array-like
        Predicted probabilities.
    cost_fp : float
        Cost of False Positive (friction, customer annoyance, manual verification, churn risk).
    cost_fn : float
        Cost of False Negative (direct fraud loss, chargeback fees, card reissuance).
    num_thresholds : int
        Number of threshold candidates to evaluate in [0.01, 0.99].
        
    Returns:
    --------
    results : dict
        optimal_threshold, min_cost, default_cost, cost_savings, and threshold_curve df.
    """
    thresholds = np.linspace(0.01, 0.99, num_thresholds)
    cost_history = []
    
    total_positives = int(np.sum(y_true == 1))
    total_negatives = int(np.sum(y_true == 0))
    
    for t in thresholds:
        preds = (y_scores >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()
        
        financial_cost = (fp * cost_fp) + (fn * cost_fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        
        cost_history.append({
            "threshold": t,
            "cost": financial_cost,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "tn": tn,
            "precision": precision,
            "recall": recall
        })
        
    df_costs = pd.DataFrame(cost_history)
    opt_idx = df_costs['cost'].idxmin()
    optimal_row = df_costs.loc[opt_idx]
    
    # Default 0.5 threshold stats
    default_row = df_costs.iloc[(df_costs['threshold'] - 0.5).abs().argsort()[:1]].iloc[0]
    
    savings = float(default_row['cost'] - optimal_row['cost'])
    savings_pct = float((savings / default_row['cost']) * 100) if default_row['cost'] > 0 else 0.0
    
    return {
        "cost_fp": cost_fp,
        "cost_fn": cost_fn,
        "optimal_threshold": float(optimal_row['threshold']),
        "optimal_cost": float(optimal_row['cost']),
        "optimal_fp": int(optimal_row['fp']),
        "optimal_fn": int(optimal_row['fn']),
        "optimal_precision": float(optimal_row['precision']),
        "optimal_recall": float(optimal_row['recall']),
        "default_threshold": 0.5,
        "default_cost": float(default_row['cost']),
        "default_fp": int(default_row['fp']),
        "default_fn": int(default_row['fn']),
        "cost_savings": savings,
        "cost_savings_pct": savings_pct,
        "cost_curve": df_costs
    }
