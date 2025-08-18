"""Tests for filename parsing and data discovery."""

import pytest
from pathlib import Path

from data_ingest import FilenameParser, DatasetDiscovery


class TestFilenameParser:
    """Test filename parsing functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.parser = FilenameParser()
    
    def test_valid_filename_parsing(self):
        """Test parsing of valid UMAFall filenames."""
        filename = "UMAFall_Subject_01ADL_Walking_1_123456.csv"
        result = self.parser.parse(filename)
        
        assert result['subject_id'] == "Subject_01"
        assert result['subject_num'] == 1
        assert result['type'] == "ADL"
        assert result['activity'] == "Walking"
        assert result['trial'] == 1
        assert result['timestamp'] == "123456"
    
    def test_fall_filename_parsing(self):
        """Test parsing of fall activity filenames."""
        filename = "UMAFall_Subject_05FALL_backwardFall_2_789012.csv"
        result = self.parser.parse(filename)
        
        assert result['subject_id'] == "Subject_05"
        assert result['type'] == "FALL"
        assert result['activity'] == "backwardFall"
        assert result['trial'] == 2
    
    def test_invalid_filename_raises_error(self):
        """Test that invalid filenames raise ValueError."""
        invalid_filenames = [
            "invalid_filename.csv",
            "UMAFall_Subject_01_Walking.csv",
            "Subject_01ADL_Walking_1_123456.csv"
        ]
        
        for filename in invalid_filenames:
            with pytest.raises(ValueError):
                self.parser.parse(filename)
    
    def test_activity_validation(self):
        """Test activity name validation."""
        # Valid ADL activity
        assert self.parser.validate_activity("Walking", "ADL")
        
        # Valid FALL activity
        assert self.parser.validate_activity("backwardFall", "FALL")
        
        # Invalid activity
        assert not self.parser.validate_activity("InvalidActivity", "ADL")
        assert not self.parser.validate_activity("Walking", "FALL")
