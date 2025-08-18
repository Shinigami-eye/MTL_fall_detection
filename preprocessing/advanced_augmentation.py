# File: preprocessing/advanced_augmentation.py (NEW)
class AdvancedAugmentation:
    """Advanced augmentation techniques for time series."""
    
    def mixup(self, x1, x2, y1, y2, alpha=0.2):
        """MixUp augmentation for time series."""
        lam = np.random.beta(alpha, alpha)
        x_mixed = lam * x1 + (1 - lam) * x2
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed
    
    def cutmix(self, x1, x2, y1, y2, alpha=1.0):
        """CutMix for time series."""
        lam = np.random.beta(alpha, alpha)
        seq_len = x1.shape[0]
        
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(0, seq_len - cut_len)
        
        x_mixed = x1.copy()
        x_mixed[cut_start:cut_start + cut_len] = x2[cut_start:cut_start + cut_len]
        
        # Mix labels proportionally
        y_mixed = lam * y1 + (1 - lam) * y2
        return x_mixed, y_mixed
    
    def spec_augment(self, x, time_mask=0.1, channel_mask=0.1):
        """SpecAugment-style masking for sensor data."""
        seq_len, n_channels = x.shape
        
        # Time masking
        mask_len = int(seq_len * time_mask)
        mask_start = np.random.randint(0, seq_len - mask_len)
        x[mask_start:mask_start + mask_len, :] = 0
        
        # Channel masking
        n_mask_channels = int(n_channels * channel_mask)
        mask_channels = np.random.choice(n_channels, n_mask_channels, replace=False)
        x[:, mask_channels] = 0
        
        return x