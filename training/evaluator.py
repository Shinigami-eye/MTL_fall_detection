"""
Evaluation metrics and analysis for multi-task fall detection model
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    precision_recall_curve, auc, roc_curve, roc_auc_score
)
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import pandas as pd
from tqdm import tqdm


class MultiTaskEvaluator:
    """
    Comprehensive evaluation for multi-task fall detection model
    """
    
    def __init__(
        self,
        model: nn.Module,
        device: torch.device,
        activity_classes: List[str],
        output_dir: Optional[Path] = None
    ):
        self.model = model
        self.device = device
        self.activity_classes = activity_classes
        self.output_dir = Path(output_dir) if output_dir else Path('.')
        
        # Risky activities that often cause false positives
        self.risky_activities = [
            'sitting_down', 'lying', 'jumping', 
            'picking_up', 'bending', 'transition'
        ]
        
        self.risky_activity_indices = [
            i for i, act in enumerate(activity_classes) 
            if act in self.risky_activities
        ]
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader,
        threshold: float = 0.5
    ) -> Dict:
        """
        Comprehensive evaluation of the model
        
        Args:
            dataloader: Test dataloader
            threshold: Fall detection threshold
        
        Returns:
            Dictionary containing all metrics
        """
        self.model.eval()
        
        all_activity_preds = []
        all_activity_labels = []
        all_fall_preds = []
        all_fall_labels = []
        all_fall_probs = []
        
        # Collect predictions
        for batch in tqdm(dataloader, desc="Evaluating"):
            inputs = batch['input'].to(self.device)
            activity_labels = batch['activity_label']
            fall_labels = batch['fall_label']
            
            # Forward pass
            outputs = self.model(inputs)
            
            # Get predictions
            activity_preds = outputs['activity_logits'].argmax(dim=1).cpu()
            fall_probs = torch.sigmoid(outputs['fall_logits']).cpu()
            
            all_activity_preds.append(activity_preds)
            all_activity_labels.append(activity_labels)
            all_fall_probs.append(fall_probs)
            all_fall_labels.append(fall_labels)
        
        # Concatenate all results
        all_activity_preds = torch.cat(all_activity_preds).numpy()
        all_activity_labels = torch.cat(all_activity_labels).numpy()
        all_fall_probs = torch.cat(all_fall_probs).numpy()
        all_fall_labels = torch.cat(all_fall_labels).numpy()
        all_fall_preds = (all_fall_probs > threshold).astype(int)
        
        # Compute metrics
        metrics = {}
        
        # Activity recognition metrics
        metrics['activity'] = self._compute_activity_metrics(
            all_activity_preds, all_activity_labels
        )
        
        # Fall detection metrics
        metrics['fall'] = self._compute_fall_metrics(
            all_fall_preds, all_fall_labels, all_fall_probs
        )
        
        # Cross-task analysis
        metrics['cross_task'] = self._analyze_cross_task_performance(
            all_activity_labels, all_activity_preds,
            all_fall_labels, all_fall_preds
        )
        
        # Generate visualizations
        self._plot_confusion_matrices(
            all_activity_preds, all_activity_labels,
            all_fall_preds, all_fall_labels
        )
        
        self._plot_pr_roc_curves(all_fall_labels, all_fall_probs)
        
        self._plot_risky_activity_analysis(
            all_activity_labels, all_fall_labels, all_fall_preds
        )
        
        return metrics
    
    def _compute_activity_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray
    ) -> Dict:
        """Compute activity recognition metrics"""
        
        # Overall accuracy
        accuracy = (preds == labels).mean()
        
        # Per-class metrics
        report = classification_report(
            labels, preds,
            target_names=self.activity_classes,
            output_dict=True,
            zero_division=0
        )
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        
        # Macro and weighted F1
        macro_f1 = report['macro avg']['f1-score']
        weighted_f1 = report['weighted avg']['f1-score']
        
        return {
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'weighted_f1': weighted_f1,
            'per_class_report': report,
            'confusion_matrix': cm
        }
    
    def _compute_fall_metrics(
        self,
        preds: np.ndarray,
        labels: np.ndarray,
        probs: np.ndarray
    ) -> Dict:
        """Compute fall detection metrics"""
        
        # Basic metrics
        tp = ((preds == 1) & (labels == 1)).sum()
        fp = ((preds == 1) & (labels == 0)).sum()
        tn = ((preds == 0) & (labels == 0)).sum()
        fn = ((preds == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        specificity = tn / (tn + fp + 1e-8)
        
        # PR curve
        pr_precision, pr_recall, pr_thresholds = precision_recall_curve(labels, probs)
        pr_auc = auc(pr_recall, pr_precision)
        
        # ROC curve
        fpr, tpr, roc_thresholds = roc_curve(labels, probs)
        roc_auc = roc_auc_score(labels, probs)
        
        # Find precision at fixed recall levels
        precision_at_95_recall = self._precision_at_recall(
            pr_precision, pr_recall, target_recall=0.95
        )
        precision_at_90_recall = self._precision_at_recall(
            pr_precision, pr_recall, target_recall=0.90
        )
        
        # Alarm rate (false alarms per hour, assuming 1 window = 2.56 seconds)
        windows_per_hour = 3600 / 2.56
        alarm_rate = fp / len(labels) * windows_per_hour
        
        return {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'specificity': specificity,
            'pr_auc': pr_auc,
            'roc_auc': roc_auc,
            'precision_at_95_recall': precision_at_95_recall,
            'precision_at_90_recall': precision_at_90_recall,
            'alarm_rate_per_hour': alarm_rate,
            'pr_curve': (pr_precision, pr_recall, pr_thresholds),
            'roc_curve': (fpr, tpr, roc_thresholds),
            'confusion_matrix': np.array([[tn, fp], [fn, tp]])
        }
    
    def _precision_at_recall(
        self,
        precision: np.ndarray,
        recall: np.ndarray,
        target_recall: float
    ) -> float:
        """Find precision at a specific recall level"""
        idx = np.where(recall >= target_recall)[0]
        if len(idx) > 0:
            return precision[idx[-1]]
        return 0.0
    
    def _analyze_cross_task_performance(
        self,
        activity_labels: np.ndarray,
        activity_preds: np.ndarray,
        fall_labels: np.ndarray,
        fall_preds: np.ndarray
    ) -> Dict:
        """Analyze how activity context affects fall detection"""
        
        results = {}
        
        # False positives by activity type
        fp_by_activity = {}
        for i, activity_name in enumerate(self.activity_classes):
            mask = activity_labels == i
            if mask.sum() > 0:
                activity_fall_preds = fall_preds[mask]
                activity_fall_labels = fall_labels[mask]
                fp_rate = ((activity_fall_preds == 1) & (activity_fall_labels == 0)).sum() / mask.sum()
                fp_by_activity[activity_name] = fp_rate
        
        results['false_positive_rate_by_activity'] = fp_by_activity
        
        # Risky activity analysis
        risky_mask = np.isin(activity_labels, self.risky_activity_indices)
        non_risky_mask = ~risky_mask
        
        # FP rate for risky vs non-risky activities
        if risky_mask.sum() > 0:
            risky_fp = ((fall_preds[risky_mask] == 1) & 
                       (fall_labels[risky_mask] == 0)).sum() / risky_mask.sum()
        else:
            risky_fp = 0
        
        if non_risky_mask.sum() > 0:
            non_risky_fp = ((fall_preds[non_risky_mask] == 1) & 
                           (fall_labels[non_risky_mask] == 0)).sum() / non_risky_mask.sum()
        else:
            non_risky_fp = 0
        
        results['risky_activities_fp_rate'] = risky_fp
        results['non_risky_activities_fp_rate'] = non_risky_fp
        results['fp_reduction_ratio'] = (risky_fp - non_risky_fp) / (risky_fp + 1e-8)
        
        # Activity prediction accuracy when fall detected
        fall_detected_mask = fall_preds == 1
        if fall_detected_mask.sum() > 0:
            activity_acc_when_fall = (activity_preds[fall_detected_mask] == 
                                     activity_labels[fall_detected_mask]).mean()
        else:
            activity_acc_when_fall = 0
        
        results['activity_accuracy_when_fall_detected'] = activity_acc_when_fall
        
        return results
    
    def _plot_confusion_matrices(
        self,
        activity_preds: np.ndarray,
        activity_labels: np.ndarray,
        fall_preds: np.ndarray,
        fall_labels: np.ndarray
    ):
        """Plot confusion matrices for both tasks"""
        
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Activity confusion matrix
        activity_cm = confusion_matrix(activity_labels, activity_preds)
        sns.heatmap(
            activity_cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            xticklabels=self.activity_classes,
            yticklabels=self.activity_classes,
            ax=axes[0],
            cbar_kws={'label': 'Count'}
        )
        axes[0].set_title('Activity Recognition Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('True')
        
        # Fall detection confusion matrix
        fall_cm = confusion_matrix(fall_labels, fall_preds)
        sns.heatmap(
            fall_cm,
            annot=True,
            fmt='d',
            cmap='Reds',
            xticklabels=['No Fall', 'Fall'],
            yticklabels=['No Fall', 'Fall'],
            ax=axes[1],
            cbar_kws={'label': 'Count'}
        )
        axes[1].set_title('Fall Detection Confusion Matrix')
        axes[1].set_xlabel('Predicted')
        axes[1].set_ylabel('True')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrices.png', dpi=150)
        plt.close()
    
    def _plot_pr_roc_curves(
        self,
        fall_labels: np.ndarray,
        fall_probs: np.ndarray
    ):
        """Plot PR and ROC curves for fall detection"""
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # PR curve
        precision, recall, _ = precision_recall_curve(fall_labels, fall_probs)
        pr_auc = auc(recall, precision)
        
        axes[0].plot(recall, precision, label=f'PR AUC = {pr_auc:.3f}')
        axes[0].axhline(y=0.85, color='r', linestyle='--', alpha=0.5, label='Target Precision')
        axes[0].axvline(x=0.95, color='g', linestyle='--', alpha=0.5, label='Target Recall')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # ROC curve
        fpr, tpr, _ = roc_curve(fall_labels, fall_probs)
        roc_auc = roc_auc_score(fall_labels, fall_probs)
        
        axes[1].plot(fpr, tpr, label=f'ROC AUC = {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5)
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'pr_roc_curves.png', dpi=150)
        plt.close()
    
    def _plot_risky_activity_analysis(
        self,
        activity_labels: np.ndarray,
        fall_labels: np.ndarray,
        fall_preds: np.ndarray
    ):
        """Analyze false positives for risky activities"""
        
        # Calculate FP rate for each activity
        fp_rates = []
        activities = []
        
        for i, activity_name in enumerate(self.activity_classes):
            mask = (activity_labels == i) & (fall_labels == 0)
            if mask.sum() > 0:
                fp_rate = fall_preds[mask].mean()
                fp_rates.append(fp_rate)
                activities.append(activity_name)
        
        # Sort by FP rate
        sorted_indices = np.argsort(fp_rates)[::-1]
        fp_rates = [fp_rates[i] for i in sorted_indices]
        activities = [activities[i] for i in sorted_indices]
        
        # Create bar plot
        fig, ax = plt.subplots(figsize=(12, 6))
        
        colors = ['red' if act in self.risky_activities else 'blue' 
                 for act in activities]
        
        bars = ax.bar(range(len(activities)), fp_rates, color=colors, alpha=0.7)
        ax.set_xticks(range(len(activities)))
        ax.set_xticklabels(activities, rotation=45, ha='right')
        ax.set_xlabel('Activity')
        ax.set_ylabel('False Positive Rate')
        ax.set_title('False Positive Rate by Activity Type')
        ax.axhline(y=0.1, color='g', linestyle='--', alpha=0.5, label='Target FP Rate')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='red', alpha=0.7, label='Risky Activities'),
            Patch(facecolor='blue', alpha=0.7, label='Normal Activities')
        ]
        ax.legend(handles=legend_elements)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'risky_activity_analysis.png', dpi=150)
        plt.close()
    
    def compare_models(
        self,
        models: Dict[str, nn.Module],
        dataloader: DataLoader
    ) -> pd.DataFrame:
        """Compare multiple models (e.g., single-task vs multi-task)"""
        
        results = []
        
        for model_name, model in models.items():
            print(f"Evaluating {model_name}...")
            self.model = model
            metrics = self.evaluate(dataloader)
            
            result = {
                'Model': model_name,
                'Activity Acc': metrics['activity']['accuracy'],
                'Activity F1': metrics['activity']['macro_f1'],
                'Fall Precision': metrics['fall']['precision'],
                'Fall Recall': metrics['fall']['recall'],
                'Fall F1': metrics['fall']['f1'],
                'PR AUC': metrics['fall']['pr_auc'],
                'Precision@0.95Recall': metrics['fall']['precision_at_95_recall'],
                'Risky FP Rate': metrics['cross_task']['risky_activities_fp_rate'],
                'FP Reduction': metrics['cross_task']['fp_reduction_ratio']
            }
            results.append(result)
        
        df = pd.DataFrame(results)
        df.to_csv(self.output_dir / 'model_comparison.csv', index=False)
        
        return df
