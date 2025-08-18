"""Data splitting utilities."""

from .cross_subject import CrossSubjectSplitter
from .leakage_validator import LeakageValidator
from .manifest_reader import ManifestReader
from .manifest_writer import ManifestWriter

__all__ = ["CrossSubjectSplitter", "ManifestWriter", "ManifestReader", "LeakageValidator"]

# ===========================
# File: splits/cross_subject.py
# ===========================
"""Cross-subject splitting strategies."""

import logging
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import KFold

logger = logging.getLogger(__name__)


class CrossSubjectSplitter:
    """Generate cross-subject train/val/test splits."""
    
    def __init__(self, n_subjects: int = 19, seed: int = 42):
        """
        Initialize cross-subject splitter.
        
        Args:
            n_subjects: Total number of subjects
            seed: Random seed for reproducibility
        """
        self.n_subjects = n_subjects
        self.seed = seed
        self.subject_ids = [f"Subject_{i:02d}" for i in range(1, n_subjects + 1)]
    
    def kfold_split(self, n_folds: int = 5) -> List[Dict[str, List[str]]]:
        """
        Generate k-fold cross-validation splits.
        
        Args:
            n_folds: Number of folds
            
        Returns:
            List of split dictionaries with train/val/test subject lists
        """
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=self.seed)
        splits = []
        
        subject_array = np.array(self.subject_ids)
        
        for fold_idx, (train_val_idx, test_idx) in enumerate(kf.split(subject_array)):
            # Split train_val into train and val (80/20)
            n_train = int(len(train_val_idx) * 0.8)
            np.random.seed(self.seed + fold_idx)
            np.random.shuffle(train_val_idx)
            
            train_idx = train_val_idx[:n_train]
            val_idx = train_val_idx[n_train:]
            
            split = {
                'fold': fold_idx,
                'train': subject_array[train_idx].tolist(),
                'val': subject_array[val_idx].tolist(),
                'test': subject_array[test_idx].tolist()
            }
            
            splits.append(split)
            
            logger.info(
                f"Fold {fold_idx}: train={len(split['train'])}, "
                f"val={len(split['val'])}, test={len(split['test'])}"
            )
        
        return splits
    
    def loso_split(self) -> List[Dict[str, List[str]]]:
        """
        Generate Leave-One-Subject-Out splits.
        
        Returns:
            List of split dictionaries with train/val/test subject lists
        """
        splits = []
        
        for test_subject in self.subject_ids:
            # Remaining subjects for train/val
            other_subjects = [s for s in self.subject_ids if s != test_subject]
            
            # Use last 20% of other subjects for validation
            n_val = max(1, int(len(other_subjects) * 0.2))
            
            # Deterministic val selection based on test subject
            np.random.seed(hash(test_subject) % (2**32))
            val_indices = np.random.choice(len(other_subjects), n_val, replace=False)
            
            val_subjects = [other_subjects[i] for i in val_indices]
            train_subjects = [s for s in other_subjects if s not in val_subjects]
            
            split = {
                'test_subject': test_subject,
                'train': train_subjects,
                'val': val_subjects,
                'test': [test_subject]
            }
            
            splits.append(split)
        
        logger.info(f"Generated {len(splits)} LOSO splits")
        return splits
    
    def stratified_split(self, activity_counts: Dict[str, Dict[str, int]],
                        test_ratio: float = 0.2,
                        val_ratio: float = 0.2) -> Dict[str, List[str]]:
        """
        Generate stratified split based on activity distribution.
        
        Args:
            activity_counts: Dict mapping subject_id to activity counts
            test_ratio: Ratio of subjects for test
            val_ratio: Ratio of subjects for validation
            
        Returns:
            Split dictionary with train/val/test subject lists
        """
        # Calculate total activities per subject
        subject_totals = {
            subject: sum(counts.values())
            for subject, counts in activity_counts.items()
        }
        
        # Sort subjects by total activities
        sorted_subjects = sorted(subject_totals.keys(),
                               key=lambda x: subject_totals[x])
        
        # Stratified assignment
        n_test = int(len(sorted_subjects) * test_ratio)
        n_val = int(len(sorted_subjects) * val_ratio)
        
        # Interleave subjects for balanced distribution
        test_subjects = sorted_subjects[::3][:n_test]
        val_subjects = sorted_subjects[1::3][:n_val]
        train_subjects = [s for s in sorted_subjects
                         if s not in test_subjects and s not in val_subjects]
        
        split = {
            'train': train_subjects,
            'val': val_subjects,
            'test': test_subjects
        }
        
        logger.info(
            f"Stratified split: train={len(train_subjects)}, "
            f"val={len(val_subjects)}, test={len(test_subjects)}"
        )
        
        return split