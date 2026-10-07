"""
Main pipeline runner for model training, evaluation, threshold optimization,
and drift monitoring.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Headless backend for reliable plot generation
import matplotlib.pyplot as plt
import seaborn as sns
from download_data import ensure_dataset
from src.data_loader import load_credit_card_data, get_imbalance_statistics, preprocess_and_split
from src.models import train_supervised_model, train_unsupervised_detector, get_anomaly_scores
from src.evaluation import evaluate_predictions, explain_imbalance_metrics, optimize_cost_threshold
from src.drift_monitor import DriftMonitor
from src.drift_simulation import run_drift_experiment

# Directories
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(PROJECT_ROOT, "results")
FIGURES_DIR = os.path.join(RESULTS_DIR, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)

def plot_pr_and_roc_curves(eval_lgb, eval_iso, save_path):
    """Generates comparison plots for PR and ROC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # 1. Precision-Recall Curve (The true indicator for extreme imbalance)
    ax = axes[0]
    ax.plot(
        eval_lgb['curves']['recall_curve'], 
        eval_lgb['curves']['precision_curve'], 
        label=f"Supervised LightGBM (PR-AUC = {eval_lgb['pr_auc']:.4f})", 
        color="#1f77b4", lw=2
    )
    ax.plot(
        eval_iso['curves']['recall_curve'], 
        eval_iso['curves']['precision_curve'], 
        label=f"Unsupervised IsolationForest (PR-AUC = {eval_iso['pr_auc']:.4f})", 
        color="#ff7f0e", lw=2, linestyle="--"
    )
    ax.axhline(y=0.0017, color='grey', linestyle=':', label='Baseline / Random Chance (0.17%)')
    ax.set_title("Precision-Recall Curve (Primary Benchmark)", fontsize=13, fontweight='bold')
    ax.set_xlabel("Recall", fontsize=11)
    ax.set_ylabel("Precision", fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.set_xlim([0.0, 1.0])
    ax.grid(True, alpha=0.3)
    ax.legend(loc='lower left', fontsize=10)
    
    # 2. ROC Curve (Shows contrast of inflated metric)
    ax2 = axes[1]
    ax2.plot(
        eval_lgb['curves']['fpr_curve'], 
        eval_lgb['curves']['tpr_curve'], 
        label=f"Supervised LightGBM (ROC-AUC = {eval_lgb['roc_auc']:.4f})", 
        color="#1f77b4", lw=2
    )
    ax2.plot(
        eval_iso['curves']['fpr_curve'], 
        eval_iso['curves']['tpr_curve'], 
        label=f"Unsupervised IsolationForest (ROC-AUC = {eval_iso['roc_auc']:.4f})", 
        color="#ff7f0e", lw=2, linestyle="--"
    )
    ax2.plot([0, 1], [0, 1], color='grey', linestyle=':', label='Random Chance (0.50)')
    ax2.set_title("ROC Curve (Noticeably Inflated by 99.8% TNs)", fontsize=13, fontweight='bold')
    ax2.set_xlabel("False Positive Rate", fontsize=11)
    ax2.set_ylabel("True Positive Rate", fontsize=11)
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlim([0.0, 1.0])
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_cost_curve(cost_results, save_path):
    """Plots the Financial Loss vs Classification Threshold curve."""
    df_cost = cost_results['cost_curve']
    opt_t = cost_results['optimal_threshold']
    opt_c = cost_results['optimal_cost']
    def_c = cost_results['default_cost']
    
    plt.figure(figsize=(9, 5))
    plt.plot(df_cost['threshold'], df_cost['cost'], color="#d62728", lw=2.5, label='Total Business Cost ($)')
    plt.axvline(x=0.5, color='gray', linestyle='--', label=f"Default Threshold (0.5) - Cost: ${def_c:,.2f}")
    plt.scatter([opt_t], [opt_c], color='green', s=100, zorder=5, label=f"Optimal Threshold ({opt_t:.2f}) - Cost: ${opt_c:,.2f}")
    
    plt.title("Financial Cost vs. Decision Threshold Optimization", fontsize=13, fontweight='bold')
    plt.xlabel("Decision Probability Threshold", fontsize=11)
    plt.ylabel("Total Expected Loss ($)", fontsize=11)
    plt.annotate(
        f"Savings: ${cost_results['cost_savings']:,.2f} ({cost_results['cost_savings_pct']:.1f}%)",
        xy=(opt_t, opt_c),
        xytext=(opt_t + 0.1, opt_c + (def_c - opt_c) * 0.4),
        arrowprops=dict(facecolor='black', shrink=0.05, width=1, headwidth=6),
        fontweight='bold',
        fontsize=10
    )
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def plot_drift_psi_distribution(drift_df, save_path):
    """Plots bar chart of top feature PSI drift values."""
    top_10 = drift_df.head(10).copy()
    
    colors = []
    for s in top_10['status']:
        if s == 'CRITICAL_DRIFT':
            colors.append('#d62728') # Red
        elif s == 'MODERATE_DRIFT':
            colors.append('#ff7f0e') # Orange
        else:
            colors.append('#2ca02c') # Green
            
    plt.figure(figsize=(10, 5))
    bars = plt.bar(top_10['feature'], top_10['psi'], color=colors, edgecolor='black', alpha=0.85)
    plt.axhline(y=0.25, color='#d62728', linestyle='--', lw=1.5, label='Critical Drift Retrain Bound (PSI=0.25)')
    plt.axhline(y=0.10, color='#ff7f0e', linestyle=':', lw=1.5, label='Warning Bound (PSI=0.10)')
    
    plt.title("Population Stability Index (PSI) Across Top Features", fontsize=13, fontweight='bold')
    plt.xlabel("Features", fontsize=11)
    plt.ylabel("PSI Value", fontsize=11)
    plt.xticks(rotation=45, ha='right')
    plt.grid(axis='y', alpha=0.3)
    plt.legend(loc='upper right', fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()

def main():
    print("=" * 80)
    print(" AIML-02: FRAUD / ANOMALY DETECTION WITH DRIFT MONITORING PIPELINE ")
    print("=" * 80)
    
    # Step 1: Ensure dataset availability
    print("\n[Step 1/7] Ingesting and Verifying Dataset...")
    data_path = ensure_dataset()
    df = load_credit_card_data(data_path)
    stats = get_imbalance_statistics(df)
    print(f" -> Total Records: {stats['total_transactions']:,}")
    print(f" -> Legitimate:    {stats['legitimate_count']:,} ({100 - stats['fraud_percentage']:.2f}%)")
    print(f" -> Fraudulent:    {stats['fraud_count']:,} ({stats['fraud_percentage']:.3f}%)")
    print(f" -> Imbalance:     1 fraud per ~{int(stats['imbalance_ratio']):,} genuine transactions")
    
    # Step 2: Preprocess & Split
    print("\n[Step 2/7] Preprocessing & Stratified Splitting...")
    data_dict = preprocess_and_split(df, test_size=0.25, random_state=42, split_strategy='stratified')
    X_train, X_val = data_dict['X_train'], data_dict['X_val']
    y_train, y_val = data_dict['y_train'], data_dict['y_val']
    print(f" -> Training Samples:   {len(X_train):,} (Frauds: {int(y_train.sum())})")
    print(f" -> Validation Samples: {len(X_val):,} (Frauds: {int(y_val.sum())})")
    
    # Step 3: Supervised Model Training (LightGBM with cost-sensitive class weights)
    print("\n[Step 3/7] Training Supervised Model (LightGBM with scale_pos_weight)...")
    lgb_model, lgb_info = train_supervised_model(X_train, y_train, model_type='lightgbm')
    lgb_val_probs = lgb_model.predict_proba(X_val)[:, 1]
    eval_lgb = evaluate_predictions(y_val, lgb_val_probs, threshold=0.5)
    print(f" -> Supervised LightGBM PR-AUC:  {eval_lgb['pr_auc']:.4f}")
    print(f" -> Supervised LightGBM ROC-AUC: {eval_lgb['roc_auc']:.4f}")
    print(f" -> Recall @ 0.5: {eval_lgb['recall']*100:.1f}%, Precision @ 0.5: {eval_lgb['precision']*100:.1f}%")
    
    # Step 4: Unsupervised Anomaly Detector (Isolation Forest)
    print("\n[Step 4/7] Training Unsupervised Detector (Isolation Forest - ZERO labels used)...")
    iso_model, iso_info = train_unsupervised_detector(X_train, contamination=0.0017)
    iso_val_scores = get_anomaly_scores(iso_model, X_val)
    eval_iso = evaluate_predictions(y_val, iso_val_scores, threshold=0.5)
    print(f" -> Unsupervised Isolation Forest PR-AUC:  {eval_iso['pr_auc']:.4f}")
    print(f" -> Unsupervised Isolation Forest ROC-AUC: {eval_iso['roc_auc']:.4f}")
    print(f" -> Recall @ 0.5: {eval_iso['recall']*100:.1f}%, Precision @ 0.5: {eval_iso['precision']*100:.1f}%")
    
    # Plot PR and ROC curves
    pr_roc_path = os.path.join(FIGURES_DIR, "model_comparison_pr_roc.png")
    plot_pr_and_roc_curves(eval_lgb, eval_iso, pr_roc_path)
    print(f" -> PR & ROC Comparison figure saved: {pr_roc_path}")
    
    # Step 5: Business Cost Matrix & Threshold Optimization
    print("\n[Step 5/7] Business Cost Matrix Analysis (FP Friction vs FN Fraud Loss)...")
    cost_fp = 10.0   # $10 manual review / customer friction cost
    cost_fn = 150.0  # $150 average fraud chargeback loss
    cost_opt = optimize_cost_threshold(y_val, lgb_val_probs, cost_fp=cost_fp, cost_fn=cost_fn)
    print(f" -> Default Threshold (0.50) Loss: ${cost_opt['default_cost']:,.2f}")
    print(f"    (FP: {cost_opt['default_fp']}, FN: {cost_opt['default_fn']})")
    print(f" -> Optimal Threshold ({cost_opt['optimal_threshold']:.2f}) Loss: ${cost_opt['optimal_cost']:,.2f}")
    print(f"    (FP: {cost_opt['optimal_fp']}, FN: {cost_opt['optimal_fn']})")
    print(f" -> Net Financial Savings: ${cost_opt['cost_savings']:,.2f} ({cost_opt['cost_savings_pct']:.1f}% reduction in business losses!)")
    
    cost_plot_path = os.path.join(FIGURES_DIR, "threshold_cost_optimization.png")
    plot_cost_curve(cost_opt, cost_plot_path)
    print(f" -> Cost Optimization figure saved: {cost_plot_path}")
    
    # Step 6: Drift Monitoring on Baseline vs Production Stream
    print("\n[Step 6/7] Drift Monitoring Pipeline (PSI & KS-Test)...")
    monitor = DriftMonitor(psi_warning=0.10, psi_critical=0.25)
    
    # Using temporal window
    temp_dict = preprocess_and_split(df, split_strategy='temporal', time_cutoff_ratio=0.70)
    ref_window = temp_dict['reference_window']
    later_window = temp_dict['later_window']
    
    feat_drift_df = monitor.monitor_features(ref_window, later_window)
    ref_preds = lgb_model.predict_proba(ref_window[data_dict['feature_names']])[:, 1]
    later_preds = lgb_model.predict_proba(later_window[data_dict['feature_names']])[:, 1]
    pred_drift = monitor.monitor_prediction_drift(ref_preds, later_preds)
    alert = monitor.evaluate_retraining_trigger(feat_drift_df, pred_drift)
    
    print(f" -> Temporal Window Drift Status: {alert['alert_status']}")
    print(f" -> Model Prediction Drift PSI:   {pred_drift['prediction_psi']:.4f} ({pred_drift['status']})")
    print(f" -> Top Feature PSI Drift:")
    for _, row in feat_drift_df.head(3).iterrows():
        print(f"    - {row['feature']}: PSI = {row['psi']:.4f}, KS Stat = {row['ks_statistic']:.4f} ({row['status']})")
        
    drift_plot_path = os.path.join(FIGURES_DIR, "temporal_drift_psi.png")
    plot_drift_psi_distribution(feat_drift_df, drift_plot_path)
    print(f" -> Drift distribution figure saved: {drift_plot_path}")
    
    # Step 7: Bonus Simulation - Artificial Concept & Covariate Drift Injection
    print("\n[Step 7/7] [BONUS] Simulating Artificial Drift Injection & Retraining Flag...")
    drift_experiment = run_drift_experiment(
        lgb_model, 
        baseline_ref=ref_window, 
        clean_target=later_window, 
        monitor=monitor,
        feature_cols=data_dict['feature_names']
    )
    clean_status = drift_experiment['clean_window']['alert']['alert_status']
    drifted_status = drift_experiment['drifted_window']['alert']['alert_status']
    drifted_reasons = drift_experiment['drifted_window']['alert']['reasons']
    
    print(f" -> Clean Production Stream Alert:   {clean_status}")
    print(f" -> Injected Drift Stream Alert:    {drifted_status}")
    print(f" -> Retraining Flag Triggered:      {drift_experiment['drifted_window']['alert']['retrain_flag']}")
    for r in drifted_reasons:
        print(f"    * TRIGGER REASON: {r}")
        
    # Export full metrics summary JSON
    summary_data = {
        "dataset_statistics": stats,
        "supervised_lightgbm": {
            "pr_auc": eval_lgb['pr_auc'],
            "roc_auc": eval_lgb['roc_auc'],
            "precision_at_0_5": eval_lgb['precision'],
            "recall_at_0_5": eval_lgb['recall'],
            "f1_at_0_5": eval_lgb['f1_score'],
            "confusion_matrix": eval_lgb['confusion_matrix']
        },
        "unsupervised_isolation_forest": {
            "pr_auc": eval_iso['pr_auc'],
            "roc_auc": eval_iso['roc_auc'],
            "precision_at_0_5": eval_iso['precision'],
            "recall_at_0_5": eval_iso['recall'],
            "f1_at_0_5": eval_iso['f1_score'],
            "confusion_matrix": eval_iso['confusion_matrix']
        },
        "cost_optimization": {
            "cost_fp": cost_opt['cost_fp'],
            "cost_fn": cost_opt['cost_fn'],
            "optimal_threshold": cost_opt['optimal_threshold'],
            "optimal_loss": cost_opt['optimal_cost'],
            "default_loss": cost_opt['default_cost'],
            "dollar_savings": cost_opt['cost_savings'],
            "percentage_savings": cost_opt['cost_savings_pct']
        },
        "drift_monitoring_alert": alert,
        "bonus_artificial_drift_experiment": {
            "clean_stream_alert": clean_status,
            "drifted_stream_alert": drifted_status,
            "retrain_triggered": drift_experiment['drifted_window']['alert']['retrain_flag'],
            "triggers": drifted_reasons
        }
    }
    
    summary_json_path = os.path.join(RESULTS_DIR, "metrics_summary.json")
    with open(summary_json_path, 'w') as f:
        json.dump(summary_data, f, indent=4)
        
    print(f"\n[Done] Pipeline executed successfully! Metrics saved to: {summary_json_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()
