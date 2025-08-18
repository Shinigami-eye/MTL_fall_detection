#!/usr/bin/env python
"""Verify that training run completed successfully."""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml


def verify_run():
    """Verify training run integrity."""
    print("=" * 60)
    print("UMAFall MTL Pipeline Verification")
    print("=" * 60)
    
    issues = []
    warnings = []
    
    # Check 1: Data directories exist
    print("\n1. Checking data directories...")
    required_dirs = ['data/raw', 'data/processed', 'data/manifests', 'data/stats']
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            issues.append(f"Missing directory: {dir_path}")
        else:
            print(f"  ✓ {dir_path} exists")
    
    # Check 2: Normalization statistics exist
    print("\n2. Checking normalization statistics...")
    stats_file = Path('data/stats/normalization.json')
    if not stats_file.exists():
        issues.append("Missing normalization statistics")
    else:
        with open(stats_file) as f:
            stats = json.load(f)
        print(f"  ✓ Statistics for {len(stats)} channels found")
    
    # Check 3: Manifest files exist
    print("\n3. Checking manifest files...")
    manifest_dir = Path('data/manifests')
    if manifest_dir.exists():
        manifests = list(manifest_dir.glob('manifest_*.csv'))
        if len(manifests) == 0:
            issues.append("No manifest files found")
        else:
            print(f"  ✓ {len(manifests)} manifest files found")
            
            # Check for subject leakage
            print("\n4. Checking for subject leakage...")
            for fold in range(5):  # Assuming 5 folds
                try:
                    train_df = pd.read_csv(manifest_dir / f'manifest_train_fold{fold}.csv')
                    val_df = pd.read_csv(manifest_dir / f'manifest_val_fold{fold}.csv')
                    test_df = pd.read_csv(manifest_dir / f'manifest_test_fold{fold}.csv')
                    
                    train_subjects = set(train_df['subject_id'].unique())
                    val_subjects = set(val_df['subject_id'].unique())
                    test_subjects = set(test_df['subject_id'].unique())
                    
                    # Check overlaps
                    if train_subjects & val_subjects:
                        issues.append(f"Fold {fold}: Subject leakage between train and val")
                    if train_subjects & test_subjects:
                        issues.append(f"Fold {fold}: Subject leakage between train and test")
                    if val_subjects & test_subjects:
                        issues.append(f"Fold {fold}: Subject leakage between val and test")
                    
                    if not any("Subject leakage" in issue for issue in issues):
                        print(f"  ✓ Fold {fold}: No subject leakage detected")
                        
                except FileNotFoundError:
                    warnings.append(f"Could not check fold {fold}")
    
    # Check 4: Class imbalance handling
    print("\n5. Checking class imbalance handling...")
    try:
        train_df = pd.read_csv(manifest_dir / 'manifest_train_fold0.csv')
        fall_ratio = train_df['fall_label'].mean()
        
        if fall_ratio < 0.01:
            issues.append(f"Extreme class imbalance: {fall_ratio:.1%} fall samples")
        elif fall_ratio < 0.05:
            warnings.append(f"Class imbalance: {fall_ratio:.1%} fall samples")
            print(f"  ⚠ Fall ratio: {fall_ratio:.1%} (consider using class weights)")
        else:
            print(f"  ✓ Fall ratio: {fall_ratio:.1%}")
    except:
        warnings.append("Could not check class balance")
    
    # Check 5: Model checkpoints
    print("\n6. Checking model checkpoints...")
    checkpoint_dir = Path('checkpoints')
    if checkpoint_dir.exists():
        checkpoints = list(checkpoint_dir.glob('**/best_model.pt'))
        if len(checkpoints) == 0:
            warnings.append("No model checkpoints found")
        else:
            print(f"  ✓ {len(checkpoints)} model checkpoints found")
    else:
        warnings.append("Checkpoint directory not found")
    
    # Check 6: Reports and metrics
    print("\n7. Checking evaluation reports...")
    reports_dir = Path('reports')
    if reports_dir.exists():
        reports = list(reports_dir.glob('**/*.json')) + list(reports_dir.glob('**/*.csv'))
        if len(reports) == 0:
            warnings.append("No evaluation reports found")
        else:
            print(f"  ✓ {len(reports)} report files found")
    
    # Summary
    print("\n" + "=" * 60)
    if issues:
        print("VERIFICATION FAILED")
        print(f"\n{len(issues)} critical issue(s) found:")
        for issue in issues:
            print(f"  ✗ {issue}")
    else:
        print("VERIFICATION PASSED")
    
    if warnings:
        print(f"\n{len(warnings)} warning(s):")
        for warning in warnings:
            print(f"  ⚠ {warning}")
    
    print("=" * 60)
    
    return len(issues) == 0


if __name__ == '__main__':
    success = verify_run()
    sys.exit(0 if success else 1)