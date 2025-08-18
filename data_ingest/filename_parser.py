"""Filename parser for UMAFall dataset."""

import re
from typing import Dict, Optional


class FilenameParser:
    """Parse UMAFall filenames to extract metadata."""
    
    # Pattern: UMAFall_Subject_[ID][Type]_[Activity]_[Trial]_[Timestamp].csv
    PATTERN = re.compile(
        r"UMAFall_Subject_(\d+)(ADL|FALL)_([^_]+)_(\d+)_(\d+)\.csv"
    )
    
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
        match = self.PATTERN.match(filename)
        
        if not match:
            raise ValueError(f"Filename '{filename}' doesn't match UMAFall pattern")
        
        subject_num, type_, activity, trial, timestamp = match.groups()
        
        return {
            'subject_id': f"Subject_{int(subject_num):02d}",
            'subject_num': int(subject_num),
            'type': type_,
            'activity': activity,
            'trial': int(trial),
            'timestamp': timestamp,
            'filename': filename
        }
    
    def validate_activity(self, activity: str, type_: str,
                         config: Optional[Dict] = None) -> bool:
        """
        Validate if activity name is expected for the given type.
        
        Args:
            activity: Activity name
            type_: Type (ADL or FALL)
            config: Optional config with expected activities
            
        Returns:
            True if activity is valid
        """
        if config is None:
            # Default expected activities
            adl_activities = {
                "Applauding", "Bending", "GoDownstairs", "GoUpstairs",
                "HandsUp", "Hopping", "Jogging", "LyingDown_OnABed",
                "MakingACall", "OpeningDoor", "Sitting_GettingUpOnAChair",
                "Walking"
            }
            fall_activities = {"backwardFall", "forwardFall", "lateralFall"}
        else:
            adl_activities = set(config.get('adl_activities', []))
            fall_activities = set(config.get('fall_activities', []))
        
        if type_ == "ADL":
            return activity in adl_activities
        elif type_ == "FALL":
            return activity in fall_activities
        else:
            return False