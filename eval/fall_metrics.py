"""Metrics for fall detection evaluation."""

import numpy as np
from sklearn.metrics import (auc, confusion_matrix, precision_recall_curve,
                           precision_score, recall_score, roc_auc_score,
                           roc_curve)


def compute_fall_metrics(predictions: np.ndarray, targets: np.ndarray,
                        threshold: float = 0.5, window_duration: float = 2.56,
                        stride: float = 1.28) -> dict:
    """
    Compute fall detection metrics.
    
    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        threshold: Decision threshold
        window_duration: Window duration in seconds
        stride: Stride between windows in seconds
        
    Returns:
        Dictionary of metrics
    """
    # Binary predictions
    binary_preds = (predictions >= threshold).astype(int)
    
    # Basic metrics
    precision = precision_score(targets, binary_preds, zero_division=0)
    recall = recall_score(targets, binary_preds, zero_division=0)
    f1 = 2 * (precision * recall) / (precision + recall + 1e-8)
    
    # PR curve and AUC
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(targets, predictions)
    pr_auc = auc(pr_recall, pr_precision)
    
    # ROC AUC
    roc_auc = roc_auc_score(targets, predictions) if len(np.unique(targets)) > 1 else 0.0
    
    # Confusion matrix
    cm = confusion_matrix(targets, binary_preds)
    tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
    
    # False alarm rate (per hour)
    # Number of false positives per hour of data
    total_windows = len(predictions)
    total_hours = (total_windows * stride) / 3600
    false_alarms_per_hour = fp / total_hours if total_hours > 0 else 0
    
    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'pr_auc': pr_auc,
        'roc_auc': roc_auc,
        'threshold': threshold,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'false_alarm_per_hour': false_alarms_per_hour,
        'pr_curve': {
            'precision': pr_precision.tolist(),
            'recall': pr_recall.tolist(),
            'thresholds': pr_thresholds.tolist()
        }
    }