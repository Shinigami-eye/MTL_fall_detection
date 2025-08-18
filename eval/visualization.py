"""Visualization utilities for evaluation."""

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                         class_names: list = None, normalize: bool = True,
                         save_path: str = None) -> plt.Figure:
    """
    Plot confusion matrix.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        class_names: List of class names
        normalize: Whether to normalize the matrix
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    cm = confusion_matrix(y_true, y_pred)
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
        fmt = '.2f'
    else:
        fmt = 'd'
    
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                ax=ax)
    
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_title('Confusion Matrix')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_pr_curve(precision: np.ndarray, recall: np.ndarray,
                 auc_score: float = None, save_path: str = None) -> plt.Figure:
    """
    Plot precision-recall curve.
    
    Args:
        precision: Precision values
        recall: Recall values
        auc_score: Area under curve score
        save_path: Path to save the figure
        
    Returns:
        Matplotlib figure
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    label = 'PR Curve'
    if auc_score is not None:
        label += f' (AUC = {auc_score:.3f})'
    
    ax.plot(recall, precision, 'b-', linewidth=2, label=label)
    ax.fill_between(recall, precision, alpha=0.2)
    
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig