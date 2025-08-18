# File: scripts/prepare_data_gdrive.py (NEW)
import os
from google.colab import drive

def mount_and_prepare_gdrive_data():
    """Mount Google Drive and prepare UMAFall data."""
    
    # Mount Google Drive
    drive.mount('/content/drive')
    
    # Navigate to UMAFall folder
    umafall_path = '/content/drive/MyDrive/UMAFall'
    
    # Check if unzipping is needed
    if os.path.exists(f"{umafall_path}/UMAFall_Dataset.zip"):
        !unzip -q "{umafall_path}/UMAFall_Dataset.zip" -d "{umafall_path}/"
    
    # Now process the data
    discovery = DatasetDiscovery(f"{umafall_path}/UMAFall_Dataset")
    files = discovery.discover_files()
    
    return files