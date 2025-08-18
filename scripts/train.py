# File: scripts/train.py (FIXED)
import pandas as pd
import numpy as np

class UMAFallDataset(Dataset):
    """Fixed dataset for loading UMAFall windows."""
    
    def __init__(self, manifest_path: Path):
        """Initialize dataset from manifest."""
        self.manifest = pd.read_csv(manifest_path)
        
    def __len__(self):
        return len(self.manifest)
    
    def __getitem__(self, idx):
        row = self.manifest.iloc[idx]
        
        # Properly load window data
        data_file = np.load(row['window_path'])
        data = data_file['data']  # Shape: (samples, channels)
        
        # Ensure correct shape for model input
        if len(data.shape) == 2:
            data = data.T  # Convert to (channels, samples)
        
        return {
            'data': torch.FloatTensor(data),
            'activity_label': int(row['activity_label']),
            'fall_label': int(row['fall_label']),
            'subject_id': row['subject_id']
        }