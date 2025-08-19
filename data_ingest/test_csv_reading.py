"""Test script to verify CSV reading with actual UMAFall files."""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from data_ingest.filename_parser_fixed import FilenameParser
from data_ingest.csv_reader_fixed import UMAFallCSVReader


def test_csv_reading(sample_file_path: str):
    """Test CSV reading with a sample file."""
    
    print("=" * 60)
    print("Testing UMAFall CSV Reading")
    print("=" * 60)
    
    file_path = Path(sample_file_path)
    
    # Test filename parsing
    print("\n1. Testing filename parsing...")
    parser = FilenameParser()
    
    try:
        metadata = parser.parse(file_path.name)
        print(f"✓ Successfully parsed filename:")
        for key, value in metadata.items():
            print(f"  {key}: {value}")
    except Exception as e:
        print(f"✗ Failed to parse filename: {e}")
        return
    
    # Test CSV reading
    print("\n2. Testing CSV reading...")
    reader = UMAFallCSVReader()
    
    try:
        df, csv_metadata = reader.read_csv(file_path)
        print(f"✓ Successfully read CSV:")
        print(f"  Shape: {df.shape}")
        print(f"  Columns: {list(df.columns)}")
        print(f"  First few rows:")
        print(df.head())
        print(f"\n  Statistics:")
        print(df.describe())
        print(f"\n  Metadata:")
        for key, value in csv_metadata.items():
            if key != 'file':
                print(f"    {key}: {value}")
    except Exception as e:
        print(f"✗ Failed to read CSV: {e}")
        
        # Try to understand the file structure
        print("\n  Attempting to diagnose file structure...")
        with open(file_path, 'r') as f:
            lines = f.readlines()
            print(f"  Total lines: {len(lines)}")
            print(f"  First 5 lines:")
            for i, line in enumerate(lines[:5]):
                print(f"    Line {i}: {line.strip()[:80]}...")


if __name__ == "__main__":
    # Test with a sample file path
    # You can modify this to point to your actual file
    sample_file = "UMAFall_Subject_18_Fall_forwardFall_1_20160529_213758.csv"
    
    if len(sys.argv) > 1:
        sample_file = sys.argv[1]
    
    test_csv_reading(sample_file)