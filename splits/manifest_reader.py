"""Read data manifests for loading splits."""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

logger = logging.getLogger(__name__)


class ManifestReader:
    """Read manifest files for data loading."""
    
    def __init__(self, manifest_dir: Path):
        """
        Initialize manifest reader.
        
        Args:
            manifest_dir: Directory containing manifests
        """
        self.manifest_dir = Path(manifest_dir)
    
    def read_manifest(self, split_name: str,
                     fold: Optional[int] = None) -> pd.DataFrame:
        """
        Read manifest for a specific split.
        
        Args:
            split_name: Name of split (train/val/test)
            fold: Optional fold number
            
        Returns:
            DataFrame with manifest data
        """
        if fold is not None:
            filename = f"manifest_{split_name}_fold{fold}.csv"
        else:
            filename = f"manifest_{split_name}.csv"
        
        manifest_path = self.manifest_dir / filename
        
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found: {manifest_path}")
        
        df = pd.read_csv(manifest_path)
        logger.info(f"Loaded {split_name} manifest with {len(df)} windows")
        
        return df
    
    def read_metadata(self, split_name: str,
                     fold: Optional[int] = None) -> Dict:
        """
        Read metadata for a specific split.
        
        Args:
            split_name: Name of split
            fold: Optional fold number
            
        Returns:
            Metadata dictionary
        """
        if fold is not None:
            filename = f"manifest_{split_name}_fold{fold}.json"
        else:
            filename = f"manifest_{split_name}.json"
        
        metadata_path = self.manifest_dir / filename
        
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)
        
        return metadata
    
    def get_available_manifests(self) -> List[str]:
        """
        Get list of available manifest files.
        
        Returns:
            List of manifest filenames
        """
        csv_files = list(self.manifest_dir.glob("manifest_*.csv"))
        return [f.name for f in csv_files]
