#!/usr/bin/env python
"""Script to prepare UMAFall dataset for training."""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from omegaconf import OmegaConf
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_ingest import DatasetDiscovery, FilenameParser, SchemaInference
from preprocessing import Normalizer, WindowGenerator
from splits import CrossSubjectSplitter, ManifestWriter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description='Prepare UMAFall dataset')
    parser.add_argument('--config', type=str, default='configs/dataset.yaml',
                       help='Path to dataset configuration')
    parser.add_argument('--n_folds', type=int, default=5,
                       help='Number of cross-validation folds')
    parser.add_argument('--window_size', type=int, default=128,
                       help='Window size in samples')
    parser.add_argument('--stride', type=int, default=64,
                       help='Stride between windows')
    parser.add_argument('--add_magnitude', action='store_true',
                       help='Add magnitude channel')
    args = parser.parse_args()
    
    # Load configuration
    config = OmegaConf.load(args.config)
    
    logger.info("=" * 60)
    logger.info("UMAFall Data Preparation")
    logger.info("=" * 60)
    
    # Step 1: Dataset Discovery
    logger.info("\n1. Discovering dataset files...")
    discovery = DatasetDiscovery(config.dataset.root_dir, args.config)
    files_by_subject = discovery.discover_files()
    
    # Validate dataset
    is_valid, issues = discovery.validate_dataset()
    if not is_valid:
        logger.warning("Dataset validation issues:")
        for issue in issues:
            logger.warning(f"  - {issue}")
    
    # Print summary
    print(discovery.generate_summary_report())
    
    # Step 2: Schema Inference
    logger.info("\n2. Inferring data schema...")
    inference = SchemaInference()
    
    # Sample a few files for schema inference
    sample_files = []
    for subject_files in list(files_by_subject.values())[:3]:
        sample_files.extend(subject_files[:2])
    
    schemas = []
    for file_path in sample_files[:5]:
        schema = inference.infer_schema(file_path)
        schemas.append(schema)
        
        is_valid, issues = inference.validate_schema(schema)
        if not is_valid:
            logger.warning(f"Schema issues for {file_path.name}:")
            for issue in issues:
                logger.warning(f"  - {issue}")
    
    # Step 3: Create Splits
    logger.info("\n3. Creating cross-subject splits...")
    splitter = CrossSubjectSplitter(n_subjects=config.dataset.num_subjects)
    
    # Generate k-fold splits
    kfold_splits = splitter.kfold_split(n_folds=args.n_folds)
    
    # Generate LOSO splits
    loso_splits = splitter.loso_split()
    
    # Step 4: Process Data and Create Windows
    logger.info("\n4. Processing data and creating windows...")
    
    parser = FilenameParser()
    window_gen = WindowGenerator(
        window_size=args.window_size,
        stride=args.stride,
        sampling_rate=config.dataset.sampling_rate
    )
    
    # Process each file
    all_windows = []
    
    for subject_id, file_paths in tqdm(files_by_subject.items(), desc="Processing subjects"):
        for file_path in file_paths:
            try:
                # Parse filename
                metadata = parser.parse(file_path.name)
                
                # Read data
                df = pd.read_csv(file_path)
                
                # TODO: Extract sensor columns based on schema
                # This is a placeholder - actual implementation needs proper column mapping
                sensor_cols = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
                if all(col in df.columns for col in sensor_cols):
                    data = df[sensor_cols].values
                else:
                    logger.warning(f"Missing sensor columns in {file_path.name}")
                    continue
                
                # Generate labels
                if metadata['type'] == 'FALL':
                    activity_label = 12  # Fall class
                    fall_label = 1
                else:
                    # Map activity name to index
                    activity_idx = config.dataset.adl_activities.index(metadata['activity'])
                    activity_label = activity_idx
                    fall_label = 0
                
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
                        'fall_label': fall_label
                    }
                )
                
                all_windows.extend(windows)
                
            except Exception as e:
                logger.error(f"Error processing {file_path}: {e}")
    
    logger.info(f"Generated {len(all_windows)} total windows")
    
    # Step 5: Normalize and Save
    logger.info("\n5. Computing normalization statistics...")
    
    # Use first fold's training data for normalization
    first_split = kfold_splits[0]
    train_subjects = set(first_split['train'])
    
    train_data = []
    for window in all_windows:
        if window['subject_id'] in train_subjects:
            train_data.append(window['data'])
    
    train_data = np.vstack(train_data)
    
    # Fit normalizer
    normalizer = Normalizer(stats_path=Path(config.dataset.stats_dir) / 'normalization.json')
    normalizer.fit(train_data, columns=sensor_cols)
    
    # Step 6: Save Windows and Manifests
    logger.info("\n6. Saving processed data and manifests...")
    
    # Save windows as numpy arrays
    processed_dir = Path(config.dataset.processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    for i, window in enumerate(tqdm(all_windows, desc="Saving windows")):
        # Normalize data
        normalized_data = normalizer.transform(window['data'], columns=sensor_cols)
        
        # Save to file
        window_path = processed_dir / f"window_{i:06d}.npz"
        np.savez_compressed(
            window_path,
            data=normalized_data,
            activity_label=window['activity_label'],
            fall_label=window['fall_label'],
            subject_id=window['subject_id'],
            activity=window['activity']
        )
        
        # Update window with path
        window['path'] = str(window_path)
    
    # Step 7: Write Manifests for Each Split
    logger.info("\n7. Writing split manifests...")
    
    manifest_writer = ManifestWriter(Path(config.dataset.manifest_dir))
    
    # Write k-fold manifests
    for fold_idx, split in enumerate(kfold_splits):
        train_windows = [w for w in all_windows if w['subject_id'] in split['train']]
        val_windows = [w for w in all_windows if w['subject_id'] in split['val']]
        test_windows = [w for w in all_windows if w['subject_id'] in split['test']]
        
        manifest_writer.write_manifest(train_windows, 'train', fold_idx)
        manifest_writer.write_manifest(val_windows, 'val', fold_idx)
        manifest_writer.write_manifest(test_windows, 'test', fold_idx)
    
    # Write split summary
    manifest_writer.write_split_summary(kfold_splits)
    
    logger.info("\n" + "=" * 60)
    logger.info("Data preparation complete!")
    logger.info(f"Processed data saved to: {config.dataset.processed_dir}")
    logger.info(f"Manifests saved to: {config.dataset.manifest_dir}")
    logger.info(f"Statistics saved to: {config.dataset.stats_dir}")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
