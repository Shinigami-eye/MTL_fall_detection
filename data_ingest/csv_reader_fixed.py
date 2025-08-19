"""Fixed CSV reader for UMAFall dataset with comment headers."""

import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class UMAFallCSVReader:
    """Read UMAFall CSV files with proper header handling."""
    
    def __init__(self):
        """Initialize CSV reader."""
        # Expected column patterns (various formats found in UMAFall)
        self.column_patterns = [
            # Pattern 1: Simple column names
            ['ax', 'ay', 'az', 'gx', 'gy', 'gz'],
            # Pattern 2: With units
            ['ax(g)', 'ay(g)', 'az(g)', 'gx(deg/s)', 'gy(deg/s)', 'gz(deg/s)'],
            # Pattern 3: Alternative naming
            ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z'],
            # Pattern 4: Numbered columns (if no header)
            ['0', '1', '2', '3', '4', '5']
        ]
    
    def read_csv(self, file_path: Path) -> Tuple[pd.DataFrame, Dict]:
        """
        Read UMAFall CSV file with proper handling of headers and comments.
        
        Args:
            file_path: Path to CSV file
            
        Returns:
            Tuple of (dataframe, metadata)
        """
        metadata = {'file': str(file_path)}
        
        # First, try to detect the structure
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            lines = f.readlines()
        
        # Find where actual data starts (skip comment lines)
        data_start_idx = 0
        header_line = None
        
        for i, line in enumerate(lines):
            line = line.strip()
            
            # Skip empty lines
            if not line:
                continue
            
            # Skip comment lines (starting with % or #)
            if line.startswith('%') or line.startswith('#'):
                # Extract metadata from comments if available
                if 'Universidad de Malaga' in line:
                    metadata['source'] = 'UMA'
                continue
            
            # Check if this looks like a header line
            if any(char.isalpha() for char in line) and ',' in line:
                header_line = line
                data_start_idx = i + 1
                break
            
            # If it's numeric data, we found the start
            if ',' in line or '\t' in line:
                # Try to parse as numbers
                try:
                    values = re.split('[,\t\s]+', line)
                    float(values[0])  # Test if first value is numeric
                    data_start_idx = i
                    break
                except:
                    continue
        
        # Now read the actual data
        try:
            # Try reading with detected settings
            if header_line:
                # Read with header
                df = pd.read_csv(
                    file_path,
                    skiprows=data_start_idx,
                    header=None,
                    comment='%',
                    sep=None,  # Auto-detect separator
                    engine='python',
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # Parse and set column names from header
                columns = self._parse_header(header_line)
                if columns and len(columns) == len(df.columns):
                    df.columns = columns
                else:
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
            else:
                # Read without header
                df = pd.read_csv(
                    file_path,
                    skiprows=data_start_idx,
                    header=None,
                    comment='%',
                    sep=None,  # Auto-detect separator
                    engine='python',
                    encoding='utf-8',
                    on_bad_lines='skip'
                )
                
                # Assign default column names
                if len(df.columns) == 6:
                    df.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
                elif len(df.columns) == 7:
                    df.columns = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
                elif len(df.columns) == 3:
                    df.columns = ['ax', 'ay', 'az']
                else:
                    df.columns = [f'col_{i}' for i in range(len(df.columns))]
            
            # Remove any remaining non-numeric rows
            df = self._clean_dataframe(df)
            
            # Extract metadata
            metadata['num_samples'] = len(df)
            metadata['num_columns'] = len(df.columns)
            metadata['columns'] = list(df.columns)
            metadata['sampling_rate'] = self._estimate_sampling_rate(df)
            
            logger.info(f"Successfully read {file_path.name}: {len(df)} samples, {len(df.columns)} columns")
            
            return df, metadata
            
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            
            # Fallback: Try alternative reading methods
            try:
                df = self._fallback_read(file_path)
                metadata['num_samples'] = len(df)
                metadata['num_columns'] = len(df.columns)
                metadata['columns'] = list(df.columns)
                
                return df, metadata
                
            except Exception as e2:
                logger.error(f"Fallback reading also failed: {e2}")
                raise
    
    def _parse_header(self, header_line: str) -> List[str]:
        """Parse header line to extract column names."""
        # Remove any comment characters
        header_line = header_line.strip().lstrip('%#').strip()
        
        # Split by common delimiters
        parts = re.split('[,\t\s]+', header_line)
        
        # Clean up column names
        columns = []
        for part in parts:
            part = part.strip()
            if part and not part.isdigit():
                columns.append(part)
        
        return columns
    
    def _clean_dataframe(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove non-numeric rows and clean data."""
        # Convert to numeric, coercing errors
        for col in df.columns:
            if col != 'timestamp' and 'time' not in col.lower():
                df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove rows with NaN values
        df = df.dropna()
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df
    
    def _estimate_sampling_rate(self, df: pd.DataFrame) -> Optional[float]:
        """Estimate sampling rate from data."""
        # Look for timestamp column
        time_cols = [col for col in df.columns if 'time' in col.lower()]
        
        if time_cols:
            timestamps = df[time_cols[0]].values
            if len(timestamps) > 10:
                # Calculate average time difference
                diffs = np.diff(timestamps[:100])  # Use first 100 samples
                avg_diff = np.median(diffs)
                
                if avg_diff > 0:
                    # Determine if timestamps are in seconds or milliseconds
                    if avg_diff > 1:  # Likely milliseconds
                        return 1000.0 / avg_diff
                    else:  # Likely seconds
                        return 1.0 / avg_diff
        
        # Default assumption
        return 50.0  # 50 Hz is common for IMU data
    
    def _fallback_read(self, file_path: Path) -> pd.DataFrame:
        """Fallback method to read problematic CSV files."""
        logger.info(f"Using fallback reader for {file_path.name}")
        
        data_rows = []
        
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                
                # Skip empty lines and comments
                if not line or line.startswith('%') or line.startswith('#'):
                    continue
                
                # Try to parse as numeric data
                try:
                    # Split by comma, tab, or multiple spaces
                    values = re.split('[,\t\s]+', line)
                    
                    # Convert to float to verify they're numbers
                    numeric_values = []
                    for val in values:
                        try:
                            numeric_values.append(float(val))
                        except:
                            pass
                    
                    # Only keep if we have at least 3 numeric values (minimum for accelerometer)
                    if len(numeric_values) >= 3:
                        data_rows.append(numeric_values)
                
                except:
                    continue
        
        if not data_rows:
            raise ValueError(f"No valid data rows found in {file_path}")
        
        # Create dataframe
        df = pd.DataFrame(data_rows)
        
        # Assign column names based on number of columns
        if len(df.columns) == 6:
            df.columns = ['ax', 'ay', 'az', 'gx', 'gy', 'gz']
        elif len(df.columns) == 7:
            df.columns = ['timestamp', 'ax', 'ay', 'az', 'gx', 'gy', 'gz']
        elif len(df.columns) == 3:
            df.columns = ['ax', 'ay', 'az']
        else:
            df.columns = [f'col_{i}' for i in range(len(df.columns))]
        
        return df