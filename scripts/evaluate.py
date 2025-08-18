#!/usr/bin/env python
"""Evaluation script for trained models."""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from eval import compute_adl_metrics, compute_fall_metrics, optimize_threshold
from eval.visualization import plot_confusion_matrix, plot_pr_curve
from models import create_model
from scripts.train import UMAFallDataset, set_seed

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained model')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--config', type=str, default=None,
                       help='Path to evaluation config (uses checkpoint config if not provided)')
    parser.add_argument('--fold', type=int, default=0,
                       help='Which fold to evaluate')
    parser.add_argument('--split', type=str, default='test',
                       help='Which split to evaluate (train/val/test)')
    parser.add_argument('--output', type=str, default='reports',
                       help='Output directory for results')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    
    # Load configuration
    if args.config:
        config = OmegaConf.load(args.config)
    else:
        config = checkpoint['config']
    
    # Set seed
    set_seed(42)
    
    logger.info("=" * 60)
    logger.info("Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Evaluating on: {args.split} split, fold {args.fold}")
    
    # Create model
    model = create_model(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Load dataset
    manifest_path = Path(config.dataset.manifest_dir) / f'manifest_{args.split}_fold{args.fold}.csv'
    dataset = UMAFallDataset(manifest_path)
    
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Collect predictions
    all_activity_preds = []
    all_activity_targets = []
    all_fall_preds = []
    all_fall_targets = []
    all_subjects = []
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating"):
            inputs = batch['data'].to(args.device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Activity predictions
            activity_preds = torch.argmax(outputs['activity'], dim=1).cpu().numpy()
            all_activity_preds.extend(activity_preds)
            all_activity_targets.extend(batch['activity_label'].numpy())
            
            # Fall predictions
            fall_preds = torch.sigmoid(outputs['fall']).squeeze().cpu().numpy()
            all_fall_preds.extend(fall_preds)
            all_fall_targets.extend(batch['fall_label'].numpy())
            
            # Subjects
            all_subjects.extend(batch['subject_id'])
    
    # Convert to arrays
    all_activity_preds = np.array(all_activity_preds)
    all_activity_targets = np.array(all_activity_targets)
    all_fall_preds = np.array(all_fall_preds)
    all_fall_targets = np.array(all_fall_targets)
    
    # Optimize threshold on validation set
    if args.split == 'val':
        logger.info("\nOptimizing fall detection threshold...")
        threshold_results = optimize_threshold(
            all_fall_preds, all_fall_targets,
            metric='f_beta', beta=2.0
        )
        optimal_threshold = threshold_results['optimal_threshold']
        logger.info(f"Optimal threshold: {optimal_threshold:.3f}")
    else:
        # Use default or previously optimized threshold
        optimal_threshold = 0.5
    
    # Compute metrics
    logger.info("\nComputing metrics...")
    
    # ADL metrics
    class_names = config.dataset.adl_activities + ['fall']
    adl_metrics = compute_adl_metrics(
        all_activity_preds, all_activity_targets,
        class_names=class_names
    )
    
    # Fall metrics
    fall_metrics = compute_fall_metrics(
        all_fall_preds, all_fall_targets,
        threshold=optimal_threshold
    )
    
    # Per-subject analysis
    subject_metrics = {}
    for subject in np.unique(all_subjects):
        mask = np.array(all_subjects) == subject
        
        subject_adl = compute_adl_metrics(
            all_activity_preds[mask],
            all_activity_targets[mask]
        )
        
        subject_fall = compute_fall_metrics(
            all_fall_preds[mask],
            all_fall_targets[mask],
            threshold=optimal_threshold
        )
        
        subject_metrics[subject] = {
            'adl_f1': subject_adl['macro_f1'],
            'fall_f1': subject_fall['f1']
        }
    
    # Save results
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save metrics
    results = {
        'checkpoint': str(args.checkpoint),
        'split': args.split,
        'fold': args.fold,
        'threshold': optimal_threshold,
        'adl_metrics': adl_metrics,
        'fall_metrics': fall_metrics,
        'subject_metrics': subject_metrics
    }
    
    with open(output_dir / f'metrics_{args.split}_fold{args.fold}.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    # Generate plots
    logger.info("\nGenerating plots...")
    
    # Confusion matrix for ADL
    fig = plot_confusion_matrix(
        all_activity_targets, all_activity_preds,
        class_names=class_names,
        save_path=output_dir / f'confusion_matrix_{args.split}.png'
    )
    
    # PR curve for fall detection
    fig = plot_pr_curve(
        fall_metrics['pr_curve']['precision'],
        fall_metrics['pr_curve']['recall'],
        auc_score=fall_metrics['pr_auc'],
        save_path=output_dir / f'pr_curve_{args.split}.png'
    )
    
    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"ADL Accuracy: {adl_metrics['accuracy']:.3f}")
    logger.info(f"ADL Macro F1: {adl_metrics['macro_f1']:.3f}")
    logger.info(f"Fall Precision: {fall_metrics['precision']:.3f}")
    logger.info(f"Fall Recall: {fall_metrics['recall']:.3f}")
    logger.info(f"Fall F1: {fall_metrics['f1']:.3f}")
    logger.info(f"Fall PR-AUC: {fall_metrics['pr_auc']:.3f}")
    logger.info(f"False Alarms/Hour: {fall_metrics['false_alarm_per_hour']:.2f}")
    logger.info("=" * 60)
    
    logger.info(f"\nResults saved to {output_dir}")


if __name__ == '__main__':
    main()
