"""Integrated data preparation script that works in both environments."""

import os
import sys
import json
import zipfile
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import numpy as np
import pandas as pd
from tqdm import tqdm

# Suppress specific warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from data_ingest import DatasetDiscovery, FilenameParser, SchemaInference
from preprocessing import Normalizer, WindowGenerator
from splits import CrossSubjectSplitter, ManifestWriter


class UMAFallDataPreparation:
    """Complete data preparation pipeline for UMAFall dataset."""
    
    def __init__(self, config_path: str = 'configs/dataset.yaml'):
        """
        Initialize data preparation pipeline.
        
        Args:
            config_path: Path to dataset configuration file
        """
        self.config_path = config_path
        self.config = self._load_config()
        self.parser = FilenameParser()
        
    def _load_config(self) -> dict:
        """Load configuration from YAML file."""
        try:
            import yaml
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not installed. Using default configuration.")
            return self._get_default_config()
        except FileNotFoundError:
            logger.warning(f"Config file {self.config_path} not found. Using defaults.")
            return self._get_default_config()
    
    def _get_default_config(self) -> dict:
        """Get default configuration."""
        return {
            'dataset': {
                'name': 'UMAFall',
                'root_dir': 'data/raw/UMAFall_Dataset',
                'processed_dir': 'data/processed',
                'manifest_dir': 'data/manifests',
                'stats_dir': 'data/stats',
                'sampling_rate': 50,
                'num_subjects': 19,
                'adl_activities': [
                    'Applauding', 'Bending', 'GoDownstairs', 'GoUpstairs',
                    'HandsUp', 'Hopping', 'Jogging', 'LyingDown_OnABed',
                    'MakingACall', 'OpeningDoor', 'Sitting_GettingUpOnAChair',
                    'Walking'
                ],
                'fall_activities': ['backwardFall', 'forwardFall', 'lateralFall'],
                'sensor_columns': {
                    'accelerometer': ['ax', 'ay', 'az'],
                    'gyroscope': ['gx', 'gy', 'gz']
                }
            }
        }
    
    def extract_dataset(self, zip_path: Path, extract_to: Path) -> bool:
        """
        Extract dataset from zip file.
        
        Args:
            zip_path: Path to zip file
            extract_to: Directory to extract to
            
        Returns:
            True if successful
        """
        if not zip_path.exists():
            logger.error(f"Zip file not found: {zip_path}")
            return False
        
        logger.info(f"Extracting {zip_path} to {extract_to}...")
        
        try:
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                # Get total size for progress bar
                total_size = sum(f.file_size for f in zip_ref.filelist)
                
                # Extract with progress
                extracted_size = 0
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Extracting") as pbar:
                    for file_info in zip_ref.filelist:
                        zip_ref.extract(file_info, extract_to)
                        extracted_size += file_info.file_size
                        pbar.update(file_info.file_size)
            
            logger.info("Extraction completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to extract: {e}")
            return False
    
    def discover_and_validate(self, data_path: Path) -> Dict[str, List[Path]]:
        """
        Discover and validate dataset files.
        
        Args:
            data_path: Path to dataset directory
            
        Returns:
            Dictionary mapping subject IDs to file paths
        """
        logger.info("Discovering dataset files...")
        discovery = DatasetDiscovery(str(data_path), self.config_path)
        
        files_by_subject = discovery.discover_files()
        is_valid, issues = discovery.validate_dataset()
        
        if not is_valid:
            logger.warning("Dataset validation issues found:")
            for issue in issues:
                logger.warning(f"  - {issue}")
        
        # Print summary
        print(discovery.generate_summary_report())
        
        return files_by_subject
    
    def process_files(self, files_by_subject: Dict[str, List[Path]],
                     window_size: int = 128, stride: int = 64,
                     add_magnitude: bool = False) -> List[Dict]:
        """
        Process all files and create windows.
        
        Args:
            files_by_subject: Dictionary of files by subject
            window_size: Window size in samples
            stride: Stride between windows
            add_magnitude: Whether to add magnitude channel
            
        Returns:
            List of all windows
        """
        window_gen = WindowGenerator(
            window_size=window_size,
            stride=stride,
            sampling_rate=self.config['dataset']['sampling_rate']
        )
        
        all_windows = []
        
        for subject_id, file_paths in tqdm(files_by_subject.items(), 
                                          desc="Processing subjects"):
            for file_path in file_paths:
                try:
                    # Parse filename
                    metadata = self.parser.parse(file_path.name)
                    
                    # Read CSV
                    df = pd.read_csv(file_path)
                    
                    # Extract sensor columns
                    sensor_cols = self._get_sensor_columns(df)
                    if sensor_cols is None:
                        logger.warning(f"Missing sensor columns in {file_path.name}")
                        continue
                    
                    data = df[sensor_cols].values
                    
                    # Add magnitude if requested
                    if add_magnitude and len(sensor_cols) >= 3:
                        mag = np.sqrt(np.sum(data[:, :3]**2, axis=1, keepdims=True))
                        data = np.hstack([data, mag])
                        sensor_cols.append('magnitude')
                    
                    # Generate labels
                    activity_label, fall_label = self._get_labels(metadata)
                    labels = np.full(len(data), activity_label)
                    
                    # Generate windows
                    windows = window_gen.generate_windows(
                        data=data,
                        labels=labels,
                        metadata={
                            'subject_id': metadata['subject_id'],
                            'activity': metadata['activity'],
                            'type': metadata['type'],
                            'trial': metadata['trial'],
                            'activity_label': activity_label,
                            'fall_label': fall_label,
                            'sensor_columns': sensor_cols
                        }
                    )
                    
                    all_windows.extend(windows)
                    
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        logger.info(f"Generated {len(all_windows)} total windows")
        return all_windows
    
    def _get_sensor_columns(self, df: pd.DataFrame) -> Optional[List[str]]:
        """Get sensor column names from dataframe."""
        expected = self.config['dataset']['sensor_columns']
        
        # Try expected columns
        acc_cols = expected['accelerometer']
        gyro_cols = expected['gyroscope']
        all_cols = acc_cols + gyro_cols
        
        if all(col in df.columns for col in all_cols):
            return all_cols
        
        # Try lowercase
        df_cols_lower = [col.lower() for col in df.columns]
        all_cols_lower = [col.lower() for col in all_cols]
        
        if all(col in df_cols_lower for col in all_cols_lower):
            # Map back to original case
            return [df.columns[df_cols_lower.index(col)] for col in all_cols_lower]
        
        return None
    
    def _get_labels(self, metadata: Dict) -> Tuple[int, int]:
        """Get activity and fall labels from metadata."""
        if metadata['type'] == 'FALL':
            activity_label = 12  # Fall class
            fall_label = 1
        else:
            # Map activity name to index
            try:
                activity_idx = self.config['dataset']['adl_activities'].index(
                    metadata['activity']
                )
                activity_label = activity_idx
                fall_label = 0
            except ValueError:
                logger.warning(f"Unknown activity: {metadata['activity']}")
                activity_label = -1
                fall_label = 0
        
        return activity_label, fall_label
    
    def save_windows_and_manifests(self, all_windows: List[Dict],
                                   splits: List[Dict]) -> None:
        """Save processed windows and create manifests."""
        processed_dir = Path(self.config['dataset']['processed_dir'])
        processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Save windows
        logger.info("Saving processed windows...")
        for i, window in enumerate(tqdm(all_windows, desc="Saving")):
            window_path = processed_dir / f"window_{i:06d}.npz"
            
            np.savez_compressed(
                window_path,
                data=window['data'],
                activity_label=window['activity_label'],
                fall_label=window['fall_label'],
                subject_id=window['subject_id'],
                activity=window['activity']
            )
            
            window['path'] = str(window_path)
        
        # Write manifests
        logger.info("Writing split manifests...")
        manifest_writer = ManifestWriter(Path(self.config['dataset']['manifest_dir']))
        
        for fold_idx, split in enumerate(splits):
            train_windows = [w for w in all_windows if w['subject_id'] in split['train']]
            val_windows = [w for w in all_windows if w['subject_id'] in split['val']]
            test_windows = [w for w in all_windows if w['subject_id'] in split['test']]
            
            manifest_writer.write_manifest(train_windows, 'train', fold_idx)
            manifest_writer.write_manifest(val_windows, 'val', fold_idx)
            manifest_writer.write_manifest(test_windows, 'test', fold_idx)
        
        manifest_writer.write_split_summary(splits)


def main():
    """Main execution function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prepare UMAFall dataset')
    parser.add_argument('--data_source', type=str, choices=['gdrive', 'local', 'zip'],
                       default='local', help='Data source type')
    parser.add_argument('--data_path', type=str, 
                       default='data/raw/UMAFall_Dataset',
                       help='Path to dataset or zip file')
    parser.add_argument('--gdrive_path', type=str,
                       default='/content/drive/MyDrive/UMAFall',
                       help='Google Drive path (for Colab)')
    parser.add_argument('--config', type=str, default='configs/dataset.yaml',
                       help='Configuration file path')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--window_size', type=int, default=128,
                       help='Window size in samples')
    parser.add_argument('--stride', type=int, default=64,
                       help='Stride between windows')
    parser.add_argument('--add_magnitude', action='store_true',
                       help='Add magnitude channel')
    args = parser.parse_args()
    
    # Initialize preparation pipeline
    prep = UMAFallDataPreparation(args.config)
    
    # Handle different data sources
    if args.data_source == 'gdrive':
        # Google Drive (Colab)
        try:
            from google.colab import drive
            drive.mount('/content/drive')
            data_path = Path(args.gdrive_path) / 'UMAFall_Dataset'
        except ImportError:
            logger.error("Not in Google Colab. Use --data_source local")
            sys.exit(1)
    
    elif args.data_source == 'zip':
        # Extract from zip
        zip_path = Path(args.data_path)
        data_path = zip_path.parent / 'UMAFall_Dataset'
        if not data_path.exists():
            prep.extract_dataset(zip_path, zip_path.parent)
    
    else:
        # Local directory
        data_path = Path(args.data_path)
    
    if not data_path.exists():
        logger.error(f"Dataset not found at {data_path}")
        sys.exit(1)
    
    # Process pipeline
    logger.info("="*60)
    logger.info("UMAFall Data Preparation Pipeline")
    logger.info("="*60)
    
    # 1. Discover files
    files_by_subject = prep.discover_and_validate(data_path)
    
    # 2. Create splits
    splitter = CrossSubjectSplitter(n_subjects=len(files_by_subject))
    splits = splitter.kfold_split(n_folds=args.n_folds)
    
    # 3. Process files and create windows
    all_windows = prep.process_files(
        files_by_subject,
        window_size=args.window_size,
        stride=args.stride,
        add_magnitude=args.add_magnitude
    )
    
    # 4. Compute normalization statistics
    train_subjects = set(splits[0]['train'])
    train_data = []
    for window in all_windows:
        if window['subject_id'] in train_subjects:
            train_data.append(window['data'])
    
    train_data = np.vstack(train_data)
    
    normalizer = Normalizer(
        stats_path=Path(prep.config['dataset']['stats_dir']) / 'normalization.json'
    )
    normalizer.fit(train_data)
    
    # 5. Normalize and save
    for window in all_windows:
        window['data'] = normalizer.transform(window['data'])
    
    prep.save_windows_and_manifests(all_windows, splits)
    
    logger.info("\n" + "="*60)
    logger.info("Data preparation complete!")
    logger.info(f"Processed data: {prep.config['dataset']['processed_dir']}")
    logger.info(f"Manifests: {prep.config['dataset']['manifest_dir']}")
    logger.info(f"Statistics: {prep.config['dataset']['stats_dir']}")
    logger.info("="*60)


if __name__ == '__main__':
    main()