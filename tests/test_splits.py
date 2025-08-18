"""Tests for data splitting and leakage detection."""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from splits import CrossSubjectSplitter, LeakageValidator


class TestCrossSubjectSplitter:
    """Test cross-subject splitting functionality."""
    
    def test_kfold_split(self):
        """Test k-fold cross-validation splits."""
        splitter = CrossSubjectSplitter(n_subjects=10, seed=42)
        splits = splitter.kfold_split(n_folds=5)
        
        assert len(splits) == 5
        
        for split in splits:
            # Check that all subjects are accounted for
            all_subjects = set(split['train'] + split['val'] + split['test'])
            assert len(all_subjects) == 10
            
            # Check no overlap between splits
            train_set = set(split['train'])
            val_set = set(split['val'])
            test_set = set(split['test'])
            
            assert len(train_set & val_set) == 0
            assert len(train_set & test_set) == 0
            assert len(val_set & test_set) == 0
    
    def test_loso_split(self):
        """Test Leave-One-Subject-Out splits."""
        splitter = CrossSubjectSplitter(n_subjects=5, seed=42)
        splits = splitter.loso_split()
        
        assert len(splits) == 5
        
        for split in splits:
            # Test set should have exactly one subject
            assert len(split['test']) == 1
            
            # Train and val should have the remaining subjects
            assert len(split['train']) + len(split['val']) == 4
            
            # No overlap
            assert split['test'][0] not in split['train']
            assert split['test'][0] not in split['val']


class TestLeakageValidator:
    """Test subject leakage detection."""
    
    def test_no_leakage_detection(self):
        """Test detection when there's no leakage."""
        # Create mock manifests with no overlap
        train_df = pd.DataFrame({
            'subject_id': ['Subject_01', 'Subject_01', 'Subject_02'],
            'activity': ['Walking', 'Running', 'Walking']
        })
        
        val_df = pd.DataFrame({
            'subject_id': ['Subject_03', 'Subject_03'],
            'activity': ['Walking', 'Running']
        })
        
        test_df = pd.DataFrame({
            'subject_id': ['Subject_04', 'Subject_04'],
            'activity': ['Walking', 'Running']
        })
        
        validator = LeakageValidator()
        is_valid, issues = validator.validate_split(train_df, val_df, test_df)
        
        assert is_valid
        assert len(issues) == 0
    
    def test_leakage_detection(self):
        """Test detection when there is subject leakage."""
        # Create manifests with overlap
        train_df = pd.DataFrame({
            'subject_id': ['Subject_01', 'Subject_02'],
            'activity': ['Walking', 'Running']
        })
        
        val_df = pd.DataFrame({
            'subject_id': ['Subject_02', 'Subject_03'],  # Subject_02 overlap
            'activity': ['Walking', 'Running']
        })
        
        test_df = pd.DataFrame({
            'subject_id': ['Subject_04'],
            'activity': ['Walking']
        })
        
        validator = LeakageValidator()
        is_valid, issues = validator.validate_split(train_df, val_df, test_df)
        
        assert not is_valid
        assert len(issues) > 0
        assert any('Subject_02' in issue for issue in issues)