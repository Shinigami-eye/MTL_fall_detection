"""Validate data splits for subject leakage."""

import logging
from typing import Dict, List, Set, Tuple

import pandas as pd

logger = logging.getLogger(__name__)


class LeakageValidator:
    """Validate that there's no subject leakage between splits."""
    
    def validate_split(self, train_manifest: pd.DataFrame,
                       val_manifest: pd.DataFrame,
                       test_manifest: pd.DataFrame) -> Tuple[bool, List[str]]:
        """
        Validate a single split for subject leakage.
        
        Args:
            train_manifest: Training manifest dataframe
            val_manifest: Validation manifest dataframe
            test_manifest: Test manifest dataframe
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Get unique subjects in each split
        train_subjects = set(train_manifest['subject_id'].unique())
        val_subjects = set(val_manifest['subject_id'].unique())
        test_subjects = set(test_manifest['subject_id'].unique())
        
        # Check for overlaps
        train_val_overlap = train_subjects & val_subjects
        train_test_overlap = train_subjects & test_subjects
        val_test_overlap = val_subjects & test_subjects
        
        if train_val_overlap:
            issues.append(
                f"Subject leakage between train and val: {train_val_overlap}"
            )
        
        if train_test_overlap:
            issues.append(
                f"Subject leakage between train and test: {train_test_overlap}"
            )
        
        if val_test_overlap:
            issues.append(
                f"Subject leakage between val and test: {val_test_overlap}"
            )
        
        # Log results
        if issues:
            for issue in issues:
                logger.error(issue)
        else:
            logger.info("No subject leakage detected")
        
        return len(issues) == 0, issues
    
    def validate_all_folds(self, manifest_dir: Path,
                          n_folds: int) -> Tuple[bool, Dict[int, List[str]]]:
        """
        Validate all folds for subject leakage.
        
        Args:
            manifest_dir: Directory containing manifests
            n_folds: Number of folds to validate
            
        Returns:
            Tuple of (all_valid, dict of fold issues)
        """
        from .manifest_reader import ManifestReader
        
        reader = ManifestReader(manifest_dir)
        all_issues = {}
        all_valid = True
        
        for fold in range(n_folds):
            try:
                train_df = reader.read_manifest('train', fold)
                val_df = reader.read_manifest('val', fold)
                test_df = reader.read_manifest('test', fold)
                
                is_valid, issues = self.validate_split(train_df, val_df, test_df)
                
                if not is_valid:
                    all_valid = False
                    all_issues[fold] = issues
                    
            except FileNotFoundError as e:
                logger.error(f"Could not validate fold {fold}: {e}")
                all_valid = False
                all_issues[fold] = [str(e)]
        
        return all_valid, all_issues
