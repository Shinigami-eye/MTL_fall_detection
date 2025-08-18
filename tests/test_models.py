"""Tests for model architectures."""

import torch
import pytest

from models import CNNBiLSTM, TCN, LiteTransformer, MTLModel
from models import ActivityHead, FallHead, create_model
from omegaconf import OmegaConf


class TestBackbones:
    """Test backbone architectures."""
    
    def test_cnn_bilstm_forward(self):
        """Test CNN-BiLSTM forward pass."""
        model = CNNBiLSTM(
            input_channels=6,
            cnn_channels=[32, 64],
            kernel_sizes=[7, 5],
            lstm_hidden=64,
            lstm_layers=2,
            dropout=0.3
        )
        
        # Create dummy input
        batch_size = 16
        channels = 6
        length = 128
        x = torch.randn(batch_size, channels, length)
        
        # Forward pass
        output = model(x)
        
        # Check output shape
        assert output.shape == (batch_size, 128)  # BiLSTM: 64 * 2
    
    def test_tcn_forward(self):
        """Test TCN forward pass."""
        model = TCN(
            input_channels=6,
            num_channels=[64, 128],
            kernel_size=7,
            dropout=0.3
        )
        
        x = torch.randn(16, 6, 128)
        output = model(x)
        
        assert output.shape == (16, 128)
    
    def test_lite_transformer_forward(self):
        """Test Lite Transformer forward pass."""
        model = LiteTransformer(
            input_channels=6,
            d_model=64,
            nhead=4,
            num_layers=2,
            dim_feedforward=256,
            dropout=0.3
        )
        
        x = torch.randn(16, 6, 128)
        output = model(x)
        
        assert output.shape == (16, 64)


class TestMTLModel:
    """Test multi-task learning model."""
    
    def test_mtl_forward(self):
        """Test MTL model forward pass."""
        # Create components
        backbone = CNNBiLSTM(
            input_channels=6,
            cnn_channels=[32],
            kernel_sizes=[7],
            lstm_hidden=32,
            lstm_layers=1
        )
        
        activity_head = ActivityHead(
            input_dim=64,  # BiLSTM output
            num_classes=13,
            hidden_dims=[32],
            dropout=0.3
        )
        
        fall_head = FallHead(
            input_dim=64,
            hidden_dims=[32],
            dropout=0.3
        )
        
        # Create MTL model
        model = MTLModel(backbone, activity_head, fall_head)
        
        # Forward pass
        x = torch.randn(8, 6, 128)
        outputs = model(x)
        
        # Check outputs
        assert 'activity' in outputs
        assert 'fall' in outputs
        assert 'features' in outputs
        
        assert outputs['activity'].shape == (8, 13)
        assert outputs['fall'].shape == (8, 1)
        assert outputs['features'].shape == (8, 64)
    
    def test_parameter_count(self):
        """Test parameter counting."""
        backbone = TCN(input_channels=6, num_channels=[32])
        activity_head = ActivityHead(32, 13, [16])
        fall_head = FallHead(32, [16])
        
        model = MTLModel(backbone, activity_head, fall_head)
        param_counts = model.get_parameter_count()
        
        assert 'backbone' in param_counts
        assert 'activity_head' in param_counts
        assert 'fall_head' in param_counts
        assert 'total' in param_counts
        
        # Total should be sum of components
        assert param_counts['total'] == (
            param_counts['backbone'] +
            param_counts['activity_head'] +
            param_counts['fall_head']
        )