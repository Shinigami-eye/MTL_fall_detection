"""Write data manifests for train/val/test splits."""

import json
import logging
from pathlib import Path
from typing import Dict, List

import pandas as pd

logger = logging.getLogger(__name__)


class ManifestWriter:
    """Write manifest files for data splits."""
    
    def __init__(self, output_dir: Path):
        """
        Initialize manifest writer.
        
        Args:
            output_dir: Directory to save manifests
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def write_manifest(self, windows: List[Dict], split_name: str,
                      fold: Optional[int] = None) -> Path:
        """
        Write manifest for a data split.
        
        Args:
            windows: List of window dictionaries
            split_name: Name of split (train/val/test)
            fold: Optional fold number
            
        Returns:
            Path to saved manifest
        """
        # Create manifest dataframe
        manifest_data = []
        
        for window in windows:
            row = {
                'window_path': window.get('path', ''),
                'subject_id': window.get('subject_id', ''),
                'activity': window.get('activity', ''),
                'type': window.get('type', ''),
                'trial': window.get('trial', 0),
                'start_idx': window.get('start_idx', 0),
                'end_idx': window.get('end_idx', 0),
                'activity_label': window.get('activity_label', -1),
                'fall_label': window.get('fall_label', 0)
            }
            manifest_data.append(row)
        
        df = pd.DataFrame(manifest_data)
        
        # Generate filename
        if fold is not None:
            filename = f"manifest_{split_name}_fold{fold}.csv"
        else:
            filename = f"manifest_{split_name}.csv"
        
        output_path = self.output_dir / filename
        
        # Save manifest
        df.to_csv(output_path, index=False)
        logger.info(f"Saved {split_name} manifest with {len(df)} windows to {output_path}")
        
        # Also save metadata
        metadata = {
            'split': split_name,
            'fold': fold,
            'num_windows': len(df),
            'num_subjects': df['subject_id'].nunique(),
            'activity_distribution': df['activity'].value_counts().to_dict(),
            'fall_ratio': df['fall_label'].mean()
        }
        
        metadata_path = output_path.with_suffix('.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return output_path
    
    def write_split_summary(self, splits: List[Dict]) -> Path:
        """
        Write summary of all splits.
        
        Args:
            splits: List of split dictionaries
            
        Returns:
            Path to summary file
        """
        summary_path = self.output_dir / "splits_summary.json"
        
        with open(summary_path, 'w') as f:
            json.dump(splits, f, indent=2)
        
        logger.info(f"Saved splits summary to {summary_path}")
        return summary_path
