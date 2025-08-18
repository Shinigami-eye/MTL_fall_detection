"""
Unit tests for multi-task fall detection model
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mtl_model import (
    MultiTaskFallDetector, ModelConfig,
    SharedCNNEncoder, TemporalEncoder
)
from models.losses import (
    FocalLoss, UncertaintyWeightedLoss, 
    MultiTaskLoss, compute_gradient_conflict
)
from data.dataset import IMUDataset, DataConfig


class TestModel:
    """Test model components"""
    
    @pytest.fixture
    def model_config(self):
        return ModelConfig(
            input_channels=8,
            window_length=256,
            cnn_channels=[32, 64],
            lstm_hidden=128,
            lstm_layers=2,
            num_activities=16,
            shared_dim=256
        )
    
    @pytest.fixture
    def sample_input(self):
        """Create sample input tensor"""
        batch_size = 16
        channels = 8
        time_steps = 256
        return torch.randn(batch_size, channels, time_steps)
    
    def test_cnn_encoder(self, model_config, sample_input):
        """Test CNN encoder forward pass"""
        encoder = SharedCNNEncoder(model_config)
        output = encoder(sample_input)
        
        # Check output shape
        batch_size = sample_input.shape[0]
        expected_time = model_config.window_length // (model_config.pool_size ** len(model_config.cnn_channels))
        expected_channels = model_config.cnn_channels[-1]
        
        assert output.shape == (batch_size, expected_time, expected_channels)
        assert not torch.isnan(output).any()
    
    def test_temporal_encoder(self, model_config, sample_input):
        """Test LSTM encoder"""
        cnn_encoder = SharedCNNEncoder(model_config)
        cnn_output = cnn_encoder(sample_input)
        
        temporal_encoder = TemporalEncoder(model_config, cnn_encoder.cnn_out_dim)
        sequence_output, final_output = temporal_encoder(cnn_output)
        
        batch_size = sample_input.shape[0]
        seq_len = cnn_output.shape[1]
        hidden_dim = model_config.lstm_hidden * 2  # Bidirectional
        
        assert sequence_output.shape == (batch_size, seq_len, hidden_dim)
        assert final_output.shape == (batch_size, hidden_dim)
        assert not torch.isnan(sequence_output).any()
        assert not torch.isnan(final_output).any()
    
    def test_full_model_forward(self, model_config, sample_input):
        """Test complete model forward pass"""
        model = MultiTaskFallDetector(model_config)
        output = model(sample_input)
        
        batch_size = sample_input.shape[0]
        
        # Check output dictionary
        assert 'activity_logits' in output
        assert 'fall_logits' in output
        assert 'shared_features' in output
        
        # Check shapes
        assert output['activity_logits'].shape == (batch_size, model_config.num_activities)
        assert output['fall_logits'].shape == (batch_size,)
        assert output['shared_features'].shape == (batch_size, model_config.shared_dim)
        
        # Check for NaN
        for key, tensor in output.items():
            assert not torch.isnan(tensor).any(), f"NaN in {key}"
    
    def test_gradient_flow(self, model_config, sample_input):
        """Test gradient flow through the model"""
        model = MultiTaskFallDetector(model_config)
        output = model(sample_input)
        
        # Create dummy targets
        batch_size = sample_input.shape[0]
        activity_targets = torch.randint(0, model_config.num_activities, (batch_size,))
        fall_targets = torch.randint(0, 2, (batch_size,)).float()
        
        # Compute losses
        activity_loss = nn.CrossEntropyLoss()(output['activity_logits'], activity_targets)
        fall_loss = nn.BCEWithLogitsLoss()(output['fall_logits'], fall_targets)
        total_loss = activity_loss + fall_loss
        
        # Backward pass
        total_loss.backward()
        
        # Check gradients exist and are not NaN
        for name, param in model.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient in {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient in {name}"


class TestLosses:
    """Test loss functions"""
    
    def test_focal_loss(self):
        """Test focal loss computation"""
        focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
        
        # Create sample data
        batch_size = 32
        inputs = torch.randn(batch_size)
        targets = torch.randint(0, 2, (batch_size,)).float()
        
        loss = focal_loss(inputs, targets)
        
        assert loss.shape == ()  # Scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)
    
    def test_uncertainty_weighted_loss(self):
        """Test uncertainty weighting"""
        uw_loss = UncertaintyWeightedLoss(num_tasks=2)
        
        # Create sample losses
        losses = {
            'activity': torch.tensor(1.5),
            'fall': torch.tensor(0.8)
        }
        
        total_loss, weights = uw_loss(losses)
        
        assert total_loss.shape == ()
        assert not torch.isnan(total_loss)
        assert 'activity' in weights
        assert 'fall' in weights
        assert weights['activity'] > 0
        assert weights['fall'] > 0
    
    def test_multi_task_loss(self):
        """Test complete multi-task loss"""
        loss_fn = MultiTaskLoss(
            num_activities=16,
            weighting_strategy='static',
            activity_weight=0.3,
            fall_weight=0.7,
            focal_loss=True
        )
        
        batch_size = 32
        predictions = {
            'activity_logits': torch.randn(batch_size, 16),
            'fall_logits': torch.randn(batch_size)
        }
        targets = {
            'activity_label': torch.randint(0, 16, (batch_size,)),
            'fall_label': torch.randint(0, 2, (batch_size,)).float()
        }
        
        loss_dict = loss_fn(predictions, targets)
        
        assert 'total_loss' in loss_dict
        assert 'activity_loss' in loss_dict
        assert 'fall_loss' in loss_dict
        assert 'weights' in loss_dict
        
        assert not torch.isnan(loss_dict['total_loss'])
        assert loss_dict['total_loss'].item() > 0
    
    def test_gradient_conflict(self):
        """Test gradient conflict computation"""
        # Create a simple model
        model = nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 2)
        )
        
        # Create two different losses
        input_data = torch.randn(16, 10)
        output = model(input_data)
        
        loss1 = output[:, 0].mean()
        loss2 = -output[:, 1].mean()  # Opposite direction
        
        conflict = compute_gradient_conflict(model, loss1, loss2)
        
        assert isinstance(conflict, float)
        assert -1 <= conflict <= 1  # Cosine similarity range
        # These losses should have negative conflict
        assert conflict < 0


class TestDataset:
    """Test dataset functionality"""
    
    @pytest.fixture
    def data_config(self):
        return DataConfig(
            window_size=2.56,
            sampling_rate=100,
            overlap=0.5
        )
    
    def test_dataset_creation(self, data_config):
        """Test dataset initialization"""
        dataset = IMUDataset(
            data_path='dummy_path',
            split='train',
            config=data_config,
            augment=False
        )
        
        assert len(dataset) > 0
        assert dataset.config.window_samples == 256
        assert dataset.config.stride_samples == 128
    
    def test_dataset_getitem(self, data_config):
        """Test dataset __getitem__"""
        dataset = IMUDataset(
            data_path='dummy_path',
            split='train',
            config=data_config
        )
        
        sample = dataset[0]
        
        assert 'input' in sample
        assert 'activity_label' in sample
        assert 'fall_label' in sample
        assert 'sample_weight' in sample
        
        # Check tensor shapes
        assert sample['input'].shape == (8, 256)  # 8 channels (6 IMU + 2 magnitude)
        assert sample['activity_label'].shape == ()
        assert sample['fall_label'].shape == ()
    
    def test_balanced_sampler(self, data_config):
        """Test balanced sampler creation"""
        dataset = IMUDataset(
            data_path='dummy_path',
            split='train',
            config=data_config
        )
        
        sampler = dataset.get_balanced_sampler()
        
        assert sampler is not None
        assert len(sampler) == len(dataset)


class TestIntegration:
    """Integration tests"""
    
    def test_end_to_end_forward(self):
        """Test complete forward pass from data to loss"""
        # Create dataset
        data_config = DataConfig()
        dataset = IMUDataset('dummy_path', 'train', data_config)
        
        # Create model
        model_config = ModelConfig()
        model = MultiTaskFallDetector(model_config)
        
        # Create loss
        loss_fn = MultiTaskLoss(
            num_activities=16,
            weighting_strategy='uncertainty'
        )
        
        # Get a batch
        batch = dataset[0]
        inputs = batch['input'].unsqueeze(0)  # Add batch dimension
        
        # Forward pass
        predictions = model(inputs)
        
        # Compute loss
        targets = {
            'activity_label': batch['activity_label'].unsqueeze(0),
            'fall_label': batch['fall_label'].unsqueeze(0)
        }
        
        loss_dict = loss_fn(predictions, targets)
        
        # Check everything works
        assert loss_dict['total_loss'].item() > 0
        assert not torch.isnan(loss_dict['total_loss'])
    
    def test_training_step(self):
        """Test a single training step"""
        # Setup
        model = MultiTaskFallDetector()
        optimizer = torch.optim.Adam(model.parameters())
        loss_fn = MultiTaskLoss()
        
        # Create dummy batch
        batch_size = 8
        inputs = torch.randn(batch_size, 8, 256)
        targets = {
            'activity_label': torch.randint(0, 16, (batch_size,)),
            'fall_label': torch.randint(0, 2, (batch_size,)).float()
        }
        
        # Training step
        model.train()
        optimizer.zero_grad()
        
        predictions = model(inputs)
        loss_dict = loss_fn(predictions, targets)
        loss_dict['total_loss'].backward()
        
        # Check gradients before step
        grad_norms = model.get_task_gradients()
        assert grad_norms['activity_grad_norm'] > 0
        assert grad_norms['fall_grad_norm'] > 0
        
        optimizer.step()
        
        # Loss should be finite
        assert loss_dict['total_loss'].item() < float('inf')


def test_reproducibility():
    """Test that setting seed produces reproducible results"""
    import random
    from run_experiment import set_seed
    
    # First run
    set_seed(42)
    model1 = MultiTaskFallDetector()
    input1 = torch.randn(4, 8, 256)
    output1 = model1(input1)
    
    # Second run with same seed
    set_seed(42)
    model2 = MultiTaskFallDetector()
    input2 = torch.randn(4, 8, 256)
    output2 = model2(input2)
    
    # Outputs should be identical
    for key in output1:
        if torch.is_tensor(output1[key]):
            assert torch.allclose(output1[key], output2[key], atol=1e-6)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
