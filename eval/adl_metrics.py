"""Metrics for ADL activity recognition evaluation."""

import numpy as np
from sklearn.metrics import (accuracy_score, classification_report,
                           confusion_matrix, f1_score)


def compute_adl_metrics(predictions: np.ndarray, targets: np.ndarray,
                       class_names: list = None) -> dict:
    """
    Compute ADL recognition metrics.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth class indices
        class_names: Optional list of class names
        
    Returns:
        Dictionary of metrics
    """
    # Accuracy
    accuracy = accuracy_score(targets, predictions)
    
    # F1 scores
    macro_f1 = f1_score(targets, predictions, average='macro', zero_division=0)
    weighted_f1 = f1_score(targets, predictions, average='weighted', zero_division=0)
    
    # Per-class F1
    per_class_f1 = f1_score(targets, predictions, average=None, zero_division=0)
    
    # Confusion matrix
    cm = confusion_matrix(targets, predictions)
    
    # Classification report
    report = classification_report(
        targets, predictions,
        target_names=class_names,
        output_dict=True,
        zero_division=0
    )
    
    # Create metrics dictionary
    metrics = {
        'accuracy': accuracy,
        'macro_f1': macro_f1,
        'weighted_f1': weighted_f1,
        'per_class_f1': per_class_f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    # Add per-class metrics
    if class_names:
        for i, class_name in enumerate(class_names):
            if i < len(per_class_f1):
                metrics[f'f1_{class_name}'] = per_class_f1[i]
    
    return metrics