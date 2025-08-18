"""Model definitions for multi-task learning."""

from .backbones import CNNBiLSTM, LiteTransformer, TCN
from .heads import ActivityHead, FallHead
from .model_factory import create_model
from .mtl_model import MTLModel

__all__ = ["MTLModel", "CNNBiLSTM", "TCN", "LiteTransformer",
           "ActivityHead", "FallHead", "create_model"]

# ===========================
# File: models/backbones.py
# ===========================
"""Backbone architectures for feature extraction."""

import math
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class CNNBiLSTM(nn.Module):
    """CNN + Bidirectional LSTM backbone."""
    
    def __init__(self, input_channels: int, cnn_channels: List[int],
                 kernel_sizes: List[int], lstm_hidden: int,
                 lstm_layers: int, dropout: float = 0.3):
        """
        Initialize CNN-BiLSTM backbone.
        
        Args:
            input_channels: Number of input channels
            cnn_channels: List of CNN channel sizes
            kernel_sizes: List of kernel sizes for CNN layers
            lstm_hidden: LSTM hidden size
            lstm_layers: Number of LSTM layers
            dropout: Dropout rate
        """
        super().__init__()
        
        # CNN layers
        cnn_layers = []
        in_channels = input_channels
        
        for out_channels, kernel_size in zip(cnn_channels, kernel_sizes):
            cnn_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(2)
            ])
            in_channels = out_channels
        
        self.cnn = nn.Sequential(*cnn_layers)
        
        # BiLSTM
        self.lstm = nn.LSTM(
            input_size=cnn_channels[-1],
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0
        )
        
        self.output_dim = lstm_hidden * 2  # Bidirectional
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Output features of shape (batch, output_dim)
        """
        # CNN feature extraction
        cnn_out = self.cnn(x)  # (batch, channels, reduced_length)
        
        # Prepare for LSTM
        cnn_out = cnn_out.transpose(1, 2)  # (batch, reduced_length, channels)
        
        # LSTM processing
        lstm_out, _ = self.lstm(cnn_out)  # (batch, reduced_length, hidden*2)
        
        # Global pooling
        features = torch.mean(lstm_out, dim=1)  # (batch, hidden*2)
        
        return features


class TCN(nn.Module):
    """Temporal Convolutional Network backbone."""
    
    def __init__(self, input_channels: int, num_channels: List[int],
                 kernel_size: int = 7, dropout: float = 0.3,
                 dilation_base: int = 2):
        """
        Initialize TCN backbone.
        
        Args:
            input_channels: Number of input channels
            num_channels: List of channel sizes for TCN blocks
            kernel_size: Kernel size for convolutions
            dropout: Dropout rate
            dilation_base: Base for exponential dilation
        """
        super().__init__()
        
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation = dilation_base ** i
            in_channels = input_channels if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            
            layers.append(
                TCNBlock(in_channels, out_channels, kernel_size,
                        dilation=dilation, dropout=dropout)
            )
        
        self.network = nn.Sequential(*layers)
        self.output_dim = num_channels[-1]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Output features of shape (batch, output_dim)
        """
        # Process through TCN blocks
        out = self.network(x)  # (batch, channels, length)
        
        # Global average pooling
        features = torch.mean(out, dim=-1)  # (batch, channels)
        
        return features


class TCNBlock(nn.Module):
    """Single TCN residual block."""
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        
        # Causal padding
        padding = (kernel_size - 1) * dilation
        
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)
        
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout2 = nn.Dropout(dropout)
        
        # Residual connection
        self.residual = nn.Conv1d(in_channels, out_channels, 1) \
            if in_channels != out_channels else nn.Identity()
        
        self.relu = nn.ReLU()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        residual = self.residual(x)
        
        # First conv block
        out = self.conv1(x)
        out = out[:, :, :-self.conv1.padding[0]]  # Remove causal padding
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)
        
        # Second conv block
        out = self.conv2(out)
        out = out[:, :, :-self.conv2.padding[0]]  # Remove causal padding
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)
        
        return self.relu(out + residual)


class LiteTransformer(nn.Module):
    """Lightweight Transformer backbone for temporal data."""
    
    def __init__(self, input_channels: int, d_model: int, nhead: int,
                 num_layers: int, dim_feedforward: int, dropout: float = 0.3,
                 max_seq_length: int = 256):
        """
        Initialize Lite Transformer backbone.
        
        Args:
            input_channels: Number of input channels
            d_model: Dimension of transformer
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Dimension of feedforward network
            dropout: Dropout rate
            max_seq_length: Maximum sequence length for positional encoding
        """
        super().__init__()
        
        # Input projection
        self.input_proj = nn.Linear(input_channels, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, dropout, max_seq_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output projection
        self.output_dim = d_model
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor of shape (batch, channels, length)
            
        Returns:
            Output features of shape (batch, d_model)
        """
        # Transpose for transformer: (batch, length, channels)
        x = x.transpose(1, 2)
        
        # Project input
        x = self.input_proj(x)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Transformer processing
        out = self.transformer(x)  # (batch, length, d_model)
        
        # Global average pooling
        features = torch.mean(out, dim=1)  # (batch, d_model)
        
        return features


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, dropout: float = 0.1,
                 max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input."""
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)