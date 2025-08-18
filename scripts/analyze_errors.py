#!/usr/bin/env python
"""Error analysis script for comparing models."""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from scipy.stats import mcnemar
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from models import create_model
from scripts.train import UMAFallDataset, set_seed

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def mcnemar_test(pred1, pred2, targets):
    """
    Perform McNemar's test to compare two models.
    
    Args:
        pred1: Binary predictions from model 1
        pred2: Binary predictions from model 2
        targets: Ground truth labels
        
    Returns:
        Dictionary with test results
    """
    # Create contingency table
    correct1 = (pred1 == targets)
    correct2 = (pred2 == targets)
    
    # Count disagreements
    n10 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
    n01 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
    
    # McNemar's test statistic
    if n10 + n01 == 0:
        return {'statistic': 0, 'p_value': 1.0, 'n10': n10, 'n01': n01}
    
    statistic = (abs(n10 - n01) - 1) ** 2 / (n10 + n01)
    
    # Chi-squared test with 1 degree of freedom
    from scipy.stats import chi2
    p_value = 1 - chi2.cdf(statistic, df=1)
    
    return {
        'statistic': statistic,
        'p_value': p_value,
        'n10': int(n10),
        'n01': int(n01),
        'significant': p_value < 0.05
    }


def main():
    parser = argparse.ArgumentParser(description='Analyze model errors')
    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to main model checkpoint')
    parser.add_argument('--baseline', type=str, default=None,
                       help='Path to baseline model for comparison')
    parser.add_argument('--fold', type=int, default=0,
                       help='Which fold to analyze')
    parser.add_argument('--output', type=str, default='reports/error_analysis',
                       help='Output directory')
    parser.add_argument('--top_k', type=int, default=100,
                       help='Number of top errors to save')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device to use')
    args = parser.parse_args()
    
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set seed
    set_seed(42)
    
    logger.info("=" * 60)
    logger.info("Error Analysis")
    logger.info("=" * 60)
    
    # Load main model
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    config = checkpoint['config']
    
    model = create_model(config.model)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(args.device)
    model.eval()
    
    # Load baseline model if provided
    baseline_model = None
    if args.baseline:
        baseline_checkpoint = torch.load(args.baseline, map_location=args.device)
        baseline_model = create_model(baseline_checkpoint['config'].model)
        baseline_model.load_state_dict(baseline_checkpoint['model_state_dict'])
        baseline_model.to(args.device)
        baseline_model.eval()
    
    # Load test dataset
    manifest_path = Path(config.dataset.manifest_dir) / f'manifest_test_fold{args.fold}.csv'
    dataset = UMAFallDataset(manifest_path)
    loader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=4)
    
    # Collect predictions and errors
    errors = []
    all_predictions = []
    baseline_predictions = [] if baseline_model else None
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(loader, desc="Analyzing")):
            inputs = batch['data'].to(args.device)
            
            # Main model predictions
            outputs = model(inputs)
            fall_probs = torch.sigmoid(outputs['fall']).squeeze().cpu().numpy()
            activity_preds = torch.argmax(outputs['activity'], dim=1).cpu().numpy()
            
            # Baseline predictions if available
            if baseline_model:
                baseline_outputs = baseline_model(inputs)
                baseline_fall = torch.sigmoid(baseline_outputs['fall']).squeeze().cpu().numpy()
                baseline_predictions.extend(baseline_fall)
            
            # Analyze errors
            for i in range(len(batch['data'])):
                fall_pred = fall_probs[i] > 0.5
                fall_true = batch['fall_label'][i].item()
                
                # Check for errors
                is_fp = fall_pred and not fall_true
                is_fn = not fall_pred and fall_true
                
                if is_fp or is_fn:
                    errors.append({
                        'batch_idx': batch_idx,
                        'sample_idx': i,
                        'error_type': 'FP' if is_fp else 'FN',
                        'fall_prob': fall_probs[i],
                        'activity_pred': activity_preds[i],
                        'activity_true': batch['activity_label'][i].item(),
                        'subject_id': batch['subject_id'][i]
                    })
                
                all_predictions.append({
                    'fall_pred': fall_pred,
                    'fall_true': fall_true,
                    'fall_prob': fall_probs[i],
                    'activity_pred': activity_preds[i],
                    'activity_true': batch['activity_label'][i].item(),
                    'subject_id': batch['subject_id'][i]
                })
    
    # Sort errors by confidence
    errors = sorted(errors, key=lambda x: abs(x['fall_prob'] - 0.5), reverse=True)
    
    # Save top errors
    top_errors = errors[:args.top_k]
    errors_df = pd.DataFrame(top_errors)
    errors_df.to_csv(output_dir / 'top_errors.csv', index=False)
    logger.info(f"Saved top {len(top_errors)} errors")
    
    # Analyze error patterns
    logger.info("\nError Pattern Analysis")
    logger.info("-" * 40)
    
    # Per-activity false positive rates
    predictions_df = pd.DataFrame(all_predictions)
    
    # Filter non-fall samples for FP analysis
    non_fall_df = predictions_df[predictions_df['fall_true'] == 0]
    
    # Calculate FP rate per activity
    fp_rates = {}
    activity_names = config.dataset.adl_activities + ['fall']
    
    for activity_idx, activity_name in enumerate(activity_names):
        activity_df = non_fall_df[non_fall_df['activity_true'] == activity_idx]
        if len(activity_df) > 0:
            fp_rate = activity_df['fall_pred'].mean()
            fp_rates[activity_name] = fp_rate
            logger.info(f"{activity_name:25} FP rate: {fp_rate:.3f}")
    
    # Plot FP rates
    fig, ax = plt.subplots(figsize=(12, 6))
    activities = list(fp_rates.keys())
    rates = list(fp_rates.values())
    
    bars = ax.bar(activities, rates)
    ax.set_xlabel('Activity')
    ax.set_ylabel('False Positive Rate')
    ax.set_title('False Positive Rates by Activity')
    ax.set_ylim([0, max(rates) * 1.2])
    
    # Highlight high FP activities
    for i, (activity, rate) in enumerate(zip(activities, rates)):
        if rate > 0.1:  # Highlight if FP rate > 10%
            bars[i].set_color('red')
        ax.text(i, rate + 0.01, f'{rate:.2f}', ha='center')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_dir / 'fp_rates_by_activity.png', dpi=150)
    
    # McNemar's test if baseline provided
    if baseline_model and baseline_predictions:
        logger.info("\nMcNemar's Test: Main vs Baseline")
        logger.info("-" * 40)
        
        main_preds = (predictions_df['fall_prob'] > 0.5).astype(int)
        baseline_preds = (np.array(baseline_predictions) > 0.5).astype(int)
        targets = predictions_df['fall_true'].values
        
        test_result = mcnemar_test(main_preds, baseline_preds, targets)
        
        logger.info(f"McNemar statistic: {test_result['statistic']:.3f}")
        logger.info(f"P-value: {test_result['p_value']:.4f}")
        logger.info(f"Main correct, Baseline wrong: {test_result['n10']}")
        logger.info(f"Main wrong, Baseline correct: {test_result['n01']}")
        
        if test_result['significant']:
            logger.info("Result: Models are significantly different (p < 0.05)")
        else:
            logger.info("Result: No significant difference between models")
    
    # Subject-wise error analysis
    subject_errors = predictions_df.groupby('subject_id').agg({
        'fall_pred': lambda x: (x != predictions_df.loc[x.index, 'fall_true']).mean()
    }).rename(columns={'fall_pred': 'error_rate'})
    
    subject_errors = subject_errors.sort_values('error_rate', ascending=False)
    subject_errors.to_csv(output_dir / 'subject_error_rates.csv')
    
    logger.info("\nSubjects with highest error rates:")
    for subject, error_rate in subject_errors.head(5).itertuples():
        logger.info(f"  {subject}: {error_rate:.3f}")
    
    logger.info(f"\nAnalysis complete. Results saved to {output_dir}")


if __name__ == '__main__':
    main()