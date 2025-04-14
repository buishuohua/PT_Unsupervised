#   -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Configuration for model
"""

from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Configuration class for model parameters"""

    # Model parameters
    max_stocks: int = 5
    feature_dim: int = 5
    d_model: int = 64
    nhead: int = 4
    num_encoder_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 1e-2
