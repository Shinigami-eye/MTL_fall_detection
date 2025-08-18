"""Schema inference for UMAFall CSV files."""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class SchemaInference:
    """Infer and validate schema of UMAFall CSV files."""
    
    def __init__(self, expected_columns: Optional[Dict[str, List[str]]] = None):
        """
        Initialize schema inference.
        
        Args:
            expected_columns: Expected sensor column names
        """
        if expected_columns is None:
            # Default expected columns
            self.expected_columns = {
                'accelerometer': ['ax', 'ay', 'az'],
                'gyroscope': ['gx', 'gy', 'gz']
            }
            # Alternative naming patterns
            self.alternatives = {
                'accelerometer': [
                    ['acc_x', 'acc_y', 'acc_z'],
                    ['a_x', 'a_y', 'a_z'],
                    ['accel_x', 'accel_y', 'accel_z']
                ],
                'gyroscope': [
                    ['gyro_x', 'gyro_y', 'gyro_z'],
                    ['g_x', 'g_y', 'g_z'],
                    ['gyr_x', 'gyr_y', 'gyr_z']
                ]
            }
        else:
            self.expected_columns = expected_columns
            self.alternatives = expected_columns.get('alternatives', {})
    
    def infer_schema(self, file_path: Path) -> Dict:
        """
        Infer schema from a CSV file.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Dictionary with schema information
        """
        # Read first few rows
        df = pd.read_csv(file_path, nrows=100)
        
        schema = {
            'file': str(file_path),
            'num_columns': len(df.columns),
            'column_names': list(df.columns),
            'column_types': {col: str(df[col].dtype) for col in df.columns},
            'shape': df.shape,
            'sensors': {},
            'sampling_rate': None
        }
        
        # Identify sensor columns
        schema['sensors'] = self._identify_sensors(df.columns)
        
        # Infer sampling rate if timestamp column exists
        if 'timestamp' in df.columns or 'time' in df.columns:
            time_col = 'timestamp' if 'timestamp' in df.columns else 'time'
            schema['sampling_rate'] = self._infer_sampling_rate(df[time_col])
        
        return schema
    
    def _identify_sensors(self, columns: List[str]) -> Dict[str, List[str]]:
        """
        Identify sensor columns from column names.
        
        Args:
            columns: List of column names
            
        Returns:
            Dictionary mapping sensor type to column names
        """
        sensors = {}
        columns_lower = [col.lower() for col in columns]
        
        # Check for expected accelerometer columns
        acc_cols = self._find_sensor_columns(
            columns, columns_lower, 'accelerometer'
        )
        if acc_cols:
            sensors['accelerometer'] = acc_cols
        
        # Check for expected gyroscope columns
        gyro_cols = self._find_sensor_columns(
            columns, columns_lower, 'gyroscope'
        )
        if gyro_cols:
            sensors['gyroscope'] = gyro_cols
        
        # Check for magnetometer (optional)
        mag_patterns = [['mx', 'my', 'mz'], ['mag_x', 'mag_y', 'mag_z']]
        for pattern in mag_patterns:
            if all(p in columns_lower for p in pattern):
                indices = [columns_lower.index(p) for p in pattern]
                sensors['magnetometer'] = [columns[i] for i in indices]
                break
        
        return sensors
    
    def _find_sensor_columns(self, columns: List[str], columns_lower: List[str],
                            sensor_type: str) -> Optional[List[str]]:
        """
        Find columns for a specific sensor type.
        
        Args:
            columns: Original column names
            columns_lower: Lowercase column names
            sensor_type: Type of sensor to find
            
        Returns:
            List of column names if found, None otherwise
        """
        # Check expected columns first
        expected = [col.lower() for col in self.expected_columns[sensor_type]]
        if all(col in columns_lower for col in expected):
            indices = [columns_lower.index(col) for col in expected]
            return [columns[i] for i in indices]
        
        # Check alternatives
        for alt_pattern in self.alternatives.get(sensor_type, []):
            alt_lower = [col.lower() for col in alt_pattern]
            if all(col in columns_lower for col in alt_lower):
                indices = [columns_lower.index(col) for col in alt_lower]
                return [columns[i] for i in indices]
        
        return None
    
    def _infer_sampling_rate(self, timestamps: pd.Series) -> Optional[float]:
        """
        Infer sampling rate from timestamps.
        
        Args:
            timestamps: Series of timestamp values
            
        Returns:
            Estimated sampling rate in Hz
        """
        if len(timestamps) < 2:
            return None
        
        # Calculate differences
        diffs = np.diff(timestamps)
        
        # Remove outliers (> 3 std from mean)
        mean_diff = np.mean(diffs)
        std_diff = np.std(diffs)
        valid_diffs = diffs[np.abs(diffs - mean_diff) < 3 * std_diff]
        
        if len(valid_diffs) == 0:
            return None
        
        # Calculate sampling rate
        avg_diff = np.mean(valid_diffs)
        
        # Assume timestamps are in milliseconds if avg_diff > 1
        if avg_diff > 1:
            sampling_rate = 1000.0 / avg_diff
        else:
            # Assume timestamps are in seconds
            sampling_rate = 1.0 / avg_diff
        
        return round(sampling_rate, 1)
    
    def validate_schema(self, schema: Dict) -> Tuple[bool, List[str]]:
        """
        Validate if schema meets requirements.
        
        Args:
            schema: Schema dictionary from infer_schema
            
        Returns:
            Tuple of (is_valid, list of issues)
        """
        issues = []
        
        # Check for required sensors
        if 'accelerometer' not in schema['sensors']:
            issues.append("Missing accelerometer columns")
        elif len(schema['sensors']['accelerometer']) != 3:
            issues.append(
                f"Expected 3 accelerometer axes, found "
                f"{len(schema['sensors']['accelerometer'])}"
            )
        
        if 'gyroscope' not in schema['sensors']:
            issues.append("Missing gyroscope columns")
        elif len(schema['sensors']['gyroscope']) != 3:
            issues.append(
                f"Expected 3 gyroscope axes, found "
                f"{len(schema['sensors']['gyroscope'])}"
            )
        
        # Check sampling rate
        if schema['sampling_rate'] is None:
            issues.append("Could not infer sampling rate")
        elif schema['sampling_rate'] < 10 or schema['sampling_rate'] > 200:
            issues.append(
                f"Unusual sampling rate: {schema['sampling_rate']} Hz "
                f"(expected 10-200 Hz)"
            )
        
        return len(issues) == 0, issues