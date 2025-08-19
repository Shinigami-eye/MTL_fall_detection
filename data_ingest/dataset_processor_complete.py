"""Complete dataset processor that handles actual UMAFall CSV structure."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

from .filename_parser_fixed import FilenameParser
from .csv_reader_fixed import UMAFallCSVReader

logger = logging.getLogger(__name__)


class UMAFallDatasetProcessor:
    """Complete processor for UMAFall dataset with proper CSV handling."""
    
    def __init__(self, root_dir: str, config: Optional[Dict] = None):
        """
        Initialize dataset processor.
        
        Args:
            root_dir: Root directory containing UMAFall data
            config: Optional configuration dictionary
        """
        self.root_dir = Path(root_dir)
        self.parser = FilenameParser()
        self.csv_reader = UMAFallCSVReader()
        self.config = config or self._get_default_config()
        
        # Track processing statistics
        self.stats = {
            'total_files': 0,
            'successful': 0,
            'failed': 0,
            'skipped': 0,
            'total_samples': 0
        }
    
    def _get_default_config(self) -> Dict:
        """Get default configuration."""
        return {
            'min_samples': 100,  # Minimum samples per file
            'max_samples': 100000,  # Maximum samples per file
            'required_columns': 3,  # Minimum columns (at least accelerometer)
            'expected_columns': ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
            'sampling_rate': 50.0
        }
    
    def process_dataset(self) -> Tuple[List[Dict], Dict]:
        """
        Process entire dataset.
        
        Returns:
            Tuple of (processed_files, statistics)
        """
        logger.info(f"Processing dataset from {self.root_dir}")
        
        # Find all CSV files
        csv_files = list(self.root_dir.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        self.stats['total_files'] = len(csv_files)
        
        processed_files = []
        
        for file_path in tqdm(csv_files, desc="Processing files"):
            try:
                # Parse filename
                file_metadata = self.parser.parse(file_path.name)
                
                # Read CSV
                df, csv_metadata = self.csv_reader.read_csv(file_path)
                
                # Validate data
                if not self._validate_data(df, csv_metadata):
                    self.stats['skipped'] += 1
                    continue
                
                # Process and store
                processed = {
                    'file_path': file_path,
                    'file_metadata': file_metadata,
                    'csv_metadata': csv_metadata,
                    'data': df,
                    'shape': df.shape,
                    'subject_id': file_metadata['subject_id'],
                    'activity': file_metadata['activity'],
                    'type': file_metadata['type'],
                    'trial': file_metadata['trial']
                }
                
                processed_files.append(processed)
                self.stats['successful'] += 1
                self.stats['total_samples'] += len(df)
                
            except Exception as e:
                logger.error(f"Failed to process {file_path.name}: {e}")
                self.stats['failed'] += 1
        
        # Generate summary
        self._print_summary()
        
        return processed_files, self.stats
    
    def _validate_data(self, df: pd.DataFrame, metadata: Dict) -> bool:
        """
        Validate that data meets requirements.
        
        Args:
            df: Data dataframe
            metadata: File metadata
            
        Returns:
            True if data is valid
        """
        # Check minimum samples
        if len(df) < self.config['min_samples']:
            logger.warning(f"File has only {len(df)} samples (min: {self.config['min_samples']})")
            return False
        
        # Check maximum samples
        if len(df) > self.config['max_samples']:
            logger.warning(f"File has {len(df)} samples (max: {self.config['max_samples']})")
            return False
        
        # Check minimum columns
        if len(df.columns) < self.config['required_columns']:
            logger.warning(f"File has only {len(df.columns)} columns (min: {self.config['required_columns']})")
            return False
        
        # Check for NaN values
        if df.isnull().any().any():
            nan_count = df.isnull().sum().sum()
            logger.warning(f"File contains {nan_count} NaN values")
            # Don't reject, but log warning
        
        # Check data ranges (basic sanity check for IMU data)
        for col in df.columns:
            if 'ax' in col or 'ay' in col or 'az' in col:
                # Accelerometer typically ±16g
                if df[col].abs().max() > 20:
                    logger.warning(f"Accelerometer column {col} has values > 20g")
            elif 'gx' in col or 'gy' in col or 'gz' in col:
                # Gyroscope typically ±2000 deg/s
                if df[col].abs().max() > 3000:
                    logger.warning(f"Gyroscope column {col} has values > 3000 deg/s")
        
        return True
    
    def _print_summary(self):
        """Print processing summary."""
        print("\n" + "=" * 60)
        print("Dataset Processing Summary")
        print("=" * 60)
        print(f"Total files found: {self.stats['total_files']}")
        print(f"Successfully processed: {self.stats['successful']}")
        print(f"Failed: {self.stats['failed']}")
        print(f"Skipped (validation): {self.stats['skipped']}")
        print(f"Total samples: {self.stats['total_samples']:,}")
        
        if self.stats['successful'] > 0:
            avg_samples = self.stats['total_samples'] / self.stats['successful']
            print(f"Average samples per file: {avg_samples:.0f}")
        
        success_rate = (self.stats['successful'] / self.stats['total_files'] * 100) if self.stats['total_files'] > 0 else 0
        print(f"Success rate: {success_rate:.1f}%")
        print("=" * 60)
    
    def save_processed_data(self, processed_files: List[Dict], output_dir: str):
        """
        Save processed data to disk.
        
        Args:
            processed_files: List of processed file dictionaries
            output_dir: Output directory
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Save individual files
        for i, pf in enumerate(tqdm(processed_files, desc="Saving processed data")):
            # Create filename
            subject = pf['subject_id']
            activity = pf['activity']
            trial = pf['trial']
            output_file = output_path / f"{subject}_{activity}_{trial}.npz"
            
            # Save as compressed numpy array
            np.savez_compressed(
                output_file,
                data=pf['data'].values,
                columns=pf['data'].columns.tolist(),
                subject_id=pf['subject_id'],
                activity=pf['activity'],
                type=pf['type'],
                trial=pf['trial'],
                shape=pf['shape']
            )
        
        # Save metadata summary
        metadata_df = pd.DataFrame([
            {
                'subject_id': pf['subject_id'],
                'activity': pf['activity'],
                'type': pf['type'],
                'trial': pf['trial'],
                'num_samples': pf['shape'][0],
                'num_columns': pf['shape'][1],
                'file_path': str(pf['file_path'])
            }
            for pf in processed_files
        ])
        
        metadata_df.to_csv(output_path / 'dataset_metadata.csv', index=False)
        logger.info(f"Saved {len(processed_files)} processed files to {output_path}")