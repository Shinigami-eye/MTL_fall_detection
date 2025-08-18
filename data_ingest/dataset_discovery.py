import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import yaml

from .filename_parser import FilenameParser

logger = logging.getLogger(__name__)


class DatasetDiscovery:
    """Discovers and inventories UMAFall dataset files."""
    
    def __init__(self, root_dir: str, config_path: Optional[str] = None):
        """
        Initialize dataset discovery.
        
        Args:
            root_dir: Root directory containing UMAFall data
            config_path: Optional path to dataset config
        """
        self.root_dir = Path(root_dir)
        self.parser = FilenameParser()
        
        # Load config if provided
        self.config = {}
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)['dataset']
    
    def discover_files(self) -> Dict[str, List[Path]]:
        """
        Discover all CSV files in the dataset.
        
        Returns:
            Dictionary mapping subject_id to list of file paths
        """
        files_by_subject = {}
        
        # Search for CSV files
        csv_files = list(self.root_dir.rglob("*.csv"))
        logger.info(f"Found {len(csv_files)} CSV files")
        
        for file_path in csv_files:
            try:
                # Parse filename
                metadata = self.parser.parse(file_path.name)
                subject_id = metadata['subject_id']
                
                if subject_id not in files_by_subject:
                    files_by_subject[subject_id] = []
                
                files_by_subject[subject_id].append(file_path)
            except ValueError as e:
                logger.warning(f"Could not parse file {file_path.name}: {e}")
        
        return files_by_subject
    
    def generate_inventory(self) -> pd.DataFrame:
        """
        Generate a detailed inventory of all files.
        
        Returns:
            DataFrame with file metadata
        """
        inventory = []
        files_by_subject = self.discover_files()
        
        for subject_id, file_paths in files_by_subject.items():
            for file_path in file_paths:
                try:
                    metadata = self.parser.parse(file_path.name)
                    metadata['file_path'] = str(file_path)
                    metadata['file_size'] = file_path.stat().st_size
                    
                    # Try to get row count
                    try:
                        df = pd.read_csv(file_path, nrows=0)
                        metadata['num_columns'] = len(df.columns)
                        metadata['column_names'] = ','.join(df.columns)
                    except:
                        metadata['num_columns'] = None
                        metadata['column_names'] = None
                    
                    inventory.append(metadata)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
        
        return pd.DataFrame(inventory)
    
    def validate_dataset(self) -> Tuple[bool, List[str]]:
        """
        Validate dataset completeness and structure.
        
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        files_by_subject = self.discover_files()
        
        # Check number of subjects
        expected_subjects = self.config.get('num_subjects', 19)
        actual_subjects = len(files_by_subject)
        
        if actual_subjects != expected_subjects:
            issues.append(
                f"Expected {expected_subjects} subjects, found {actual_subjects}"
            )
        
        # Check for expected activities
        expected_adl = set(self.config.get('adl_activities', []))
        expected_fall = set(self.config.get('fall_activities', []))
        
        found_adl = set()
        found_fall = set()
        
        inventory = self.generate_inventory()
        for _, row in inventory.iterrows():
            if row['type'] == 'ADL':
                found_adl.add(row['activity'])
            elif row['type'] == 'FALL':
                found_fall.add(row['activity'])
        
        missing_adl = expected_adl - found_adl
        missing_fall = expected_fall - found_fall
        
        if missing_adl:
            issues.append(f"Missing ADL activities: {missing_adl}")
        if missing_fall:
            issues.append(f"Missing FALL activities: {missing_fall}")
        
        return len(issues) == 0, issues
    
    def generate_summary_report(self) -> str:
        """
        Generate a summary report of the dataset.
        
        Returns:
            Formatted summary report string
        """
        inventory = self.generate_inventory()
        files_by_subject = self.discover_files()
        
        report = []
        report.append("=" * 60)
        report.append("UMAFall Dataset Summary")
        report.append("=" * 60)
        report.append(f"Total subjects: {len(files_by_subject)}")
        report.append(f"Total files: {len(inventory)}")
        report.append("")
        
        # Activity breakdown
        report.append("Activity Distribution:")
        activity_counts = inventory.groupby(['type', 'activity']).size()
        for (type_, activity), count in activity_counts.items():
            report.append(f"  {type_:4} - {activity:25} : {count:3} files")
        
        report.append("")
        
        # Subject breakdown
        report.append("Files per subject:")
        for subject_id in sorted(files_by_subject.keys()):
            count = len(files_by_subject[subject_id])
            report.append(f"  {subject_id}: {count} files")
        
        return "\n".join(report)
