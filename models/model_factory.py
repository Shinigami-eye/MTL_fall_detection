"""Factory for creating models from configuration."""

from typing import Dict

import torch.nn as nn
from omegaconf import DictConfig

from .backbones import CNNBiLSTM, LiteTransformer, TCN
from .heads import ActivityHead, FallHead
from .mtl_model import MTLModel


def create_backbone(config: DictConfig) -> nn.Module:
    """
    Create backbone from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        Backbone module
    """
    backbone_type = config.backbone
    input_channels = config.input_channels
    
    if config.use_magnitude:
        input_channels += 1  # Add magnitude channel
    
    if backbone_type == "cnn_bilstm":
        return CNNBiLSTM(
            input_channels=input_channels,
            cnn_channels=config.cnn_bilstm.cnn_channels,
            kernel_sizes=config.cnn_bilstm.kernel_sizes,
            lstm_hidden=config.cnn_bilstm.lstm_hidden,
            lstm_layers=config.cnn_bilstm.lstm_layers,
            dropout=config.trunk.dropout
        )
    
    elif backbone_type == "tcn":
        return TCN(
            input_channels=input_channels,
            num_channels=config.tcn.num_channels,
            kernel_size=config.tcn.kernel_size,
            dropout=config.tcn.dropout,
            dilation_base=config.tcn.dilation_base
        )
    
    elif backbone_type == "lite_transformer":
        return LiteTransformer(
            input_channels=input_channels,
            d_model=config.lite_transformer.d_model,
            nhead=config.lite_transformer.nhead,
            num_layers=config.lite_transformer.num_layers,
            dim_feedforward=config.lite_transformer.dim_feedforward,
            dropout=config.lite_transformer.dropout,
            max_seq_length=config.lite_transformer.max_seq_length
        )
    
    else:
        raise ValueError(f"Unknown backbone type: {backbone_type}")


def create_model(config: DictConfig) -> MTLModel:
    """
    Create complete MTL model from configuration.
    
    Args:
        config: Model configuration
        
    Returns:
        MTL model
    """
    # Create backbone
    backbone = create_backbone(config)
    
    # Get backbone output dimension
    if hasattr(backbone, 'output_dim'):
        backbone_dim = backbone.output_dim
    else:
        raise ValueError("Backbone must have 'output_dim' attribute")
    
    # Create task heads
    activity_head = ActivityHead(
        input_dim=backbone_dim,
        num_classes=config.activity_head.num_classes,
        hidden_dims=config.activity_head.hidden_dims,
        dropout=config.activity_head.dropout
    )
    
    fall_head = FallHead(
        input_dim=backbone_dim,
        hidden_dims=config.fall_head.hidden_dims,
        dropout=config.fall_head.dropout
    )
    
    # Create MTL model
    model = MTLModel(backbone, activity_head, fall_head)
    
    return model