"""Fixed filename parser for actual UMAFall dataset structure."""

import re
from typing import Dict, Optional


class FilenameParser:
    """Parse UMAFall filenames to extract metadata."""
    
    # FIXED Pattern: UMAFall_Subject_[ID]_[Type]_[Activity]_[Trial]_[Timestamp].csv
    # Note the underscores between components
    PATTERNS = [
    # Pattern for ADL or Fall with proper timestamp format
    re.compile(
        r"UMAFall_Subject_(\d+)_(ADL|Fall)_([A-Za-z]+)_(\d+)_(\d{8}_\d{6})\.csv"
    ),
    # Fallback pattern (handles ADL/FALL variations)
    re.compile(
        r"UMAFall_Subject_(\d+)_(ADL|FALL)_([A-Za-z]+)_(\d+)_(\d{8}_\d{6})\.csv"
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
                subject_num, type_, activity, trial, timestamp = match.groups()

                return {
                    'subject_id': f"Subject_{int(subject_num):02d}",
                    'subject_num': int(subject_num),
                    'type': type_.upper(),
                    'activity': activity,
                    'trial': int(trial),
                    'timestamp': timestamp,
                    'filename': filename
                }

        
        # If no pattern matches
        raise ValueError(f"Filename '{filename}' doesn't match UMAFall pattern")