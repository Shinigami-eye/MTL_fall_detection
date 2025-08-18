# File: inference/streaming.py (NEW)
class StreamingInference:
    """Real-time streaming inference for deployment."""
    
    def __init__(self, model_path, window_size=128, stride=64):
        self.model = torch.jit.load(model_path)
        self.model.eval()
        
        self.window_size = window_size
        self.stride = stride
        self.buffer = []
        
    def process_sample(self, sample):
        """Process a single sensor sample."""
        self.buffer.append(sample)
        
        # Check if we have enough samples for a window
        if len(self.buffer) >= self.window_size:
            # Extract window
            window = np.array(self.buffer[-self.window_size:])
            
            # Normalize (using pre-computed stats)
            window = self.normalize(window)
            
            # Predict
            with torch.no_grad():
                input_tensor = torch.FloatTensor(window.T).unsqueeze(0)
                output = self.model(input_tensor)
                
                fall_prob = torch.sigmoid(output['fall']).item()
                activity = torch.argmax(output['activity']).item()
            
            # Slide buffer
            self.buffer = self.buffer[self.stride:]
            
            return {
                'fall_probability': fall_prob,
                'activity': activity,
                'timestamp': time.time()
            }
        
        return None