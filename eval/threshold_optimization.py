"""Threshold optimization for fall detection."""

import numpy as np
from sklearn.metrics import fbeta_score


def optimize_threshold(predictions: np.ndarray, targets: np.ndarray,
                      metric: str = 'f_beta', beta: float = 2.0,
                      search_points: int = 100) -> dict:
    """
    Optimize decision threshold for fall detection.
    
    Args:
        predictions: Predicted probabilities
        targets: Ground truth labels
        metric: Metric to optimize ('f_beta', 'f1', 'precision', 'recall')
        beta: Beta value for F-beta score
        search_points: Number of threshold points to search
        
    Returns:
        Dictionary with optimal threshold and metrics
    """
    # Generate threshold candidates
    thresholds = np.linspace(0.01, 0.99, search_points)
    
    best_threshold = 0.5
    best_score = -1
    metrics_at_thresholds = []
    
    for threshold in thresholds:
        binary_preds = (predictions >= threshold).astype(int)
        
        if metric == 'f_beta':
            score = fbeta_score(targets, binary_preds, beta=beta, zero_division=0)
        elif metric == 'f1':
            score = fbeta_score(targets, binary_preds, beta=1.0, zero_division=0)
        elif metric == 'precision':
            from sklearn.metrics import precision_score
            score = precision_score(targets, binary_preds, zero_division=0)
        elif metric == 'recall':
            from sklearn.metrics import recall_score
            score = recall_score(targets, binary_preds, zero_division=0)
        else:
            raise ValueError(f"Unknown metric: {metric}")
        
        metrics_at_thresholds.append({
            'threshold': threshold,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    # Compute final metrics at optimal threshold
    from .fall_metrics import compute_fall_metrics
    final_metrics = compute_fall_metrics(predictions, targets, best_threshold)
    
    return {
        'optimal_threshold': best_threshold,
        'optimal_score': best_score,
        'metric_optimized': metric,
        'beta': beta if metric == 'f_beta' else None,
        'search_results': metrics_at_thresholds,
        'final_metrics': final_metrics
    }
