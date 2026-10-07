"""
Distribution and prediction drift monitoring using PSI and two-sample KS tests.
"""

import numpy as np
import pandas as pd
from scipy import stats

def calculate_psi(reference, target, num_buckets=10, epsilon=1e-4):
    """
    Computes the Population Stability Index (PSI) between a baseline
    reference distribution and an incoming target/production distribution.
    
    Formula:
        PSI = sum( (Actual% - Expected%) * ln(Actual% / Expected%) )
        
    Benchmarks:
        PSI < 0.10: No significant distribution change (Stable)
        0.10 <= PSI < 0.25: Moderate shift; monitoring advised (Warning)
        PSI >= 0.25: Significant change; action/retraining required (Critical)
    """
    reference = np.asarray(reference).ravel()
    target = np.asarray(target).ravel()
    
    # Remove NaN values if present
    reference = reference[~np.isnan(reference)]
    target = target[~np.isnan(target)]
    
    if len(reference) == 0 or len(target) == 0:
        return 0.0
        
    # Calculate quantile bin edges from reference distribution
    percentiles = np.linspace(0, 100, num_buckets + 1)
    bin_edges = np.percentile(reference, percentiles)
    
    # Ensure strictly increasing bins to handle repeated values
    bin_edges = np.unique(bin_edges)
    if len(bin_edges) < 2:
        return 0.0
    bin_edges[0] = -np.inf
    bin_edges[-1] = np.inf
    
    # Bin counts
    ref_counts, _ = np.histogram(reference, bins=bin_edges)
    tar_counts, _ = np.histogram(target, bins=bin_edges)
    
    # Proportions with Laplace smoothing epsilon
    ref_pct = (ref_counts + epsilon) / (np.sum(ref_counts) + epsilon * len(ref_counts))
    tar_pct = (tar_counts + epsilon) / (np.sum(tar_counts) + epsilon * len(tar_counts))
    
    # PSI sum
    psi_value = np.sum((tar_pct - ref_pct) * np.log(tar_pct / ref_pct))
    return float(psi_value)

def calculate_ks_test(reference, target):
    """
    Performs the two-sample Kolmogorov-Smirnov test to quantify differences
    in empirical cumulative distributions (ECDF).
    
    Returns:
    --------
    statistic : float (maximum vertical divergence D between ECDFs)
    p_value : float
    """
    res = stats.ks_2samp(reference, target)
    return float(res.statistic), float(res.pvalue)

class DriftMonitor:
    """
    Multi-feature and Model Confidence Drift Monitoring System.
    """
    def __init__(self, psi_warning=0.10, psi_critical=0.25, ks_alpha=0.01):
        self.psi_warning = psi_warning
        self.psi_critical = psi_critical
        self.ks_alpha = ks_alpha

    def monitor_features(self, ref_df, target_df, feature_cols=None):
        """
        Calculates PSI and KS statistics across all specified features.
        
        Returns:
        --------
        pd.DataFrame summarizing feature-level drift metrics and status.
        """
        if feature_cols is None:
            feature_cols = ref_df.columns.tolist()
            
        results = []
        for col in feature_cols:
            if col not in ref_df.columns or col not in target_df.columns:
                continue
                
            ref_vals = ref_df[col].values
            tar_vals = target_df[col].values
            
            psi = calculate_psi(ref_vals, tar_vals)
            ks_stat, ks_pval = calculate_ks_test(ref_vals, tar_vals)
            
            if psi >= self.psi_critical:
                status = "CRITICAL_DRIFT"
            elif psi >= self.psi_warning:
                status = "MODERATE_DRIFT"
            else:
                status = "STABLE"
                
            results.append({
                "feature": col,
                "psi": psi,
                "ks_statistic": ks_stat,
                "ks_pvalue": ks_pval,
                "status": status,
                "ref_mean": float(np.mean(ref_vals)),
                "tar_mean": float(np.mean(tar_vals)),
                "ref_std": float(np.std(ref_vals)),
                "tar_std": float(np.std(tar_vals))
            })
            
        return pd.DataFrame(results).sort_values(by="psi", ascending=False).reset_index(drop=True)

    def monitor_prediction_drift(self, ref_scores, target_scores):
        """
        Detects drift in model confidence and output probabilities.
        """
        psi = calculate_psi(ref_scores, target_scores)
        ks_stat, ks_pval = calculate_ks_test(ref_scores, target_scores)
        
        status = "CRITICAL_DRIFT" if psi >= self.psi_critical else (
            "MODERATE_DRIFT" if psi >= self.psi_warning else "STABLE"
        )
        
        return {
            "prediction_psi": psi,
            "prediction_ks_stat": ks_stat,
            "prediction_ks_pvalue": ks_pval,
            "status": status,
            "ref_mean_confidence": float(np.mean(ref_scores)),
            "target_mean_confidence": float(np.mean(target_scores))
        }

    def evaluate_retraining_trigger(self, feature_drift_df, pred_drift_dict, critical_feature_ratio=0.15):
        """
        Evaluates alerting threshold logic:
        Flags for automated retraining when:
        1. Prediction output PSI >= psi_critical, OR
        2. Ratio of critically drifted features exceeds threshold, OR
        3. Any primary sensitive feature crosses severe PSI bound.
        """
        total_features = len(feature_drift_df)
        critical_count = int(np.sum(feature_drift_df['status'] == 'CRITICAL_DRIFT'))
        warning_count = int(np.sum(feature_drift_df['status'] == 'MODERATE_DRIFT'))
        
        critical_ratio = critical_count / max(1, total_features)
        pred_critical = pred_drift_dict['status'] == 'CRITICAL_DRIFT'
        
        retrain_required = False
        reasons = []
        
        if pred_critical:
            retrain_required = True
            reasons.append(
                f"Model prediction probability distribution shifted critically (PSI = {pred_drift_dict['prediction_psi']:.3f} >= {self.psi_critical})"
            )
            
        if critical_ratio >= critical_feature_ratio:
            retrain_required = True
            reasons.append(
                f"{critical_count}/{total_features} features ({critical_ratio*100:.1f}%) exhibit critical drift exceeding boundary of {critical_feature_ratio*100:.0f}%"
            )
            
        top_drifted = feature_drift_df.iloc[0] if len(feature_drift_df) > 0 else None
        if top_drifted is not None and top_drifted['psi'] >= 0.40:
            retrain_required = True
            reasons.append(
                f"Feature '{top_drifted['feature']}' suffered extreme drift (PSI = {top_drifted['psi']:.3f})"
            )
            
        alert_status = "RETRAIN_ALERT" if retrain_required else (
            "WARNING_ALERT" if (warning_count > 0 or pred_drift_dict['status'] == 'MODERATE_DRIFT') else "HEALTHY"
        )
        
        return {
            "alert_status": alert_status,
            "retrain_flag": retrain_required,
            "reasons": reasons,
            "summary": {
                "total_features": total_features,
                "critical_drift_features": critical_count,
                "warning_drift_features": warning_count,
                "prediction_psi": pred_drift_dict['prediction_psi']
            }
        }
