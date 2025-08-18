"""Preprocessing module for UMAFall data."""

from .augmentation import DataAugmentation
from .filtering import SignalFilter
from .normalization import Normalizer
from .windowing import WindowGenerator

__all__ = ["Normalizer", "WindowGenerator", "SignalFilter", "DataAugmentation"]

# ===========================
# File: preprocessing/normalization.py
# ===========================
"""Normalization utilities for sensor data."""

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Normalizer:
    """Z-score normalization for sensor data."""
    
    def __init__(self, stats_path: Optional[Path] = None):
        """
        Initialize normalizer.
        
        Args:
            stats_path: Path to save/load normalization statistics
        """
        self.stats_path = stats_path
        self.stats: Dict[str, Dict[str, float]] = {}
        self.fitted = False
    
    def fit(self, data: Union[pd.DataFrame, np.ndarray],
            columns: Optional[List[str]] = None) -> 'Normalizer':
        """
        Compute normalization statistics from training data.
        
        Args:
            data: Training data (DataFrame or array)
            columns: Column names if data is array
            
        Returns:
            Self for chaining
        """
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            data_array = data.values
        else:
            data_array = data
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
        
        # Compute statistics per column
        for i, col in enumerate(columns):
            col_data = data_array[:, i]
            # Remove NaN values
            valid_data = col_data[~np.isnan(col_data)]
            
            if len(valid_data) > 0:
                self.stats[col] = {
                    'mean': float(np.mean(valid_data)),
                    'std': float(np.std(valid_data) + 1e-8)  # Add epsilon
                }
            else:
                logger.warning(f"Column {col} has no valid data")
                self.stats[col] = {'mean': 0.0, 'std': 1.0}
        
        self.fitted = True
        
        # Save statistics if path provided
        if self.stats_path:
            self.save_stats()
        
        logger.info(f"Fitted normalizer on {len(columns)} columns")
        return self
    
    def transform(self, data: Union[pd.DataFrame, np.ndarray],
                  columns: Optional[List[str]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Apply normalization to data.
        
        Args:
            data: Data to normalize
            columns: Column names if data is array
            
        Returns:
            Normalized data in same format as input
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted. Call fit() first or load_stats().")
        
        return_df = isinstance(data, pd.DataFrame)
        
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            data_array = data.values.copy()
        else:
            data_array = data.copy()
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
        
        # Apply normalization
        for i, col in enumerate(columns):
            if col in self.stats:
                mean = self.stats[col]['mean']
                std = self.stats[col]['std']
                data_array[:, i] = (data_array[:, i] - mean) / std
            else:
                logger.warning(f"No statistics for column {col}, skipping")
        
        if return_df:
            return pd.DataFrame(data_array, columns=columns, index=data.index)
        else:
            return data_array
    
    def fit_transform(self, data: Union[pd.DataFrame, np.ndarray],
                      columns: Optional[List[str]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Fit and transform in one step.
        
        Args:
            data: Data to fit and transform
            columns: Column names if data is array
            
        Returns:
            Normalized data
        """
        self.fit(data, columns)
        return self.transform(data, columns)
    
    def inverse_transform(self, data: Union[pd.DataFrame, np.ndarray],
                         columns: Optional[List[str]] = None) -> Union[pd.DataFrame, np.ndarray]:
        """
        Reverse normalization.
        
        Args:
            data: Normalized data
            columns: Column names if data is array
            
        Returns:
            Original scale data
        """
        if not self.fitted:
            raise ValueError("Normalizer not fitted.")
        
        return_df = isinstance(data, pd.DataFrame)
        
        if isinstance(data, pd.DataFrame):
            columns = data.columns
            data_array = data.values.copy()
        else:
            data_array = data.copy()
            if columns is None:
                columns = [f"col_{i}" for i in range(data.shape[1])]
        
        # Reverse normalization
        for i, col in enumerate(columns):
            if col in self.stats:
                mean = self.stats[col]['mean']
                std = self.stats[col]['std']
                data_array[:, i] = data_array[:, i] * std + mean
        
        if return_df:
            return pd.DataFrame(data_array, columns=columns, index=data.index)
        else:
            return data_array
    
    def save_stats(self, path: Optional[Path] = None):
        """
        Save normalization statistics to file.
        
        Args:
            path: Path to save to (uses self.stats_path if not provided)
        """
        save_path = path or self.stats_path
        if save_path is None:
            raise ValueError("No save path provided")
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(self.stats, f, indent=2)
        
        logger.info(f"Saved normalization statistics to {save_path}")
    
    def load_stats(self, path: Optional[Path] = None):
        """
        Load normalization statistics from file.
        
        Args:
            path: Path to load from (uses self.stats_path if not provided)
        """
        load_path = path or self.stats_path
        if load_path is None:
            raise ValueError("No load path provided")
        
        load_path = Path(load_path)
        
        with open(load_path, 'r') as f:
            self.stats = json.load(f)
        
        self.fitted = True
        logger.info(f"Loaded normalization statistics from {load_path}")
