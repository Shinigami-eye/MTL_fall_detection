"""Fixed filename parser for actual UMAFall dataset structure."""

import re
from typing import Dict, Optional


class FilenameParser:
    """Parse UMAFall filenames to extract metadata."""
    
    # FIXED Pattern: UMAFall_Subject_[ID]_[Type]_[Activity]_[Trial]_[Timestamp].csv
    # Note the underscores between components
    PATTERNS = [
        # Pattern 1: With underscores (actual format)
        re.compile(
            r"UMAFall_Subject_(\d+)_(ADL|Fall)_([^_]+)_(\d+)_(\d+)\.csv"
        ),
        # Pattern 2: Without underscores (fallback)
        re.compile(
            r"UMAFall_Subject_(\d+)(ADL|FALL)_([^_]+)_(\d+)_(\d+)\.csv"
        )
    ]
    
    def parse(self, filename: str) -> Dict[str, str]:
        """
        Parse a UMAFall filename.
        
        Args:
            filename: Name of the CSV file
            
        Returns:
            Dictionary with parsed metadata
            
        Raises:
            ValueError: If filename doesn't match expected pattern
        """
        # Try each pattern
        for pattern in self.PATTERNS:
            match = pattern.match(filename)
            if match:
                subject_num, type_, activity, trial, timestamp = match.groups()
                
                # Normalize type to uppercase
                type_ = type_.upper() if type_.upper() in ['ADL', 'FALL'] else type_
                
                return {
                    'subject_id': f"Subject_{int(subject_num):02d}",
                    'subject_num': int(subject_num),
                    'type': type_,
                    'activity': activity,
                    'trial': int(trial),
                    'timestamp': timestamp,
                    'filename': filename
                }
        
        # If no pattern matches
        raise ValueError(f"Filename '{filename}' doesn't match UMAFall pattern")