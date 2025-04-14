# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/2025
    @ Description: Configuration for training
"""

from dataclasses import dataclass
from typing import Optional
import torch
import datetime
import os


@dataclass
class TrainConfig:
    """Configuration class for training parameters"""

    seed: int = 42

    # Training parameters
    batch_size: int = 256
    num_epochs: int = 1000
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4

    # Optimizer parameters
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8

    # Learning rate scheduler
    scheduler_patience: int = 50
    scheduler_factor: float = 0.5
    min_lr: float = 1e-6

    # Early stopping
    patience: int = 100
    early_stopping: bool = True

    # Device configuration
    device: torch.device = torch.device(
        'cuda') if torch.cuda.is_available() else torch.device('cpu')

    save_freq: int = 10

    # Logging and checkpoints
    log_interval: int = 100
    experiment_dir: str = 'experiments'
    runs_dir: str = "runs"
    experiment_name: str = f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model_save_dir: str = "checkpoints"
    log_dir: str = "logs"

    # Continue training settings
    continue_training: bool = True
    # Format: "YYYYMMDD_HHMMSS" or None
    continue_from_experiment: Optional[str] = None

    @property
    def experiment_path(self) -> str:
        """Get the full experiment path"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        if self.continue_training and self.continue_from_experiment:
            return os.path.join(base_dir, self.experiment_dir, self.runs_dir, self.continue_from_experiment)
        return os.path.join(base_dir, self.experiment_dir, self.runs_dir, self.experiment_name)

    @property
    def checkpoint_path(self) -> str:
        """Get the checkpoint path"""
        return os.path.join(self.experiment_path, self.model_save_dir)

    def get_latest_checkpoint(self) -> Optional[str]:
        """Get the path to the latest checkpoint if it exists"""
        if not os.path.exists(self.checkpoint_path):
            return None

        checkpoints = [f for f in os.listdir(self.checkpoint_path)
                       if f.startswith('checkpoint_') and f.endswith('.pth')
                       and not f.startswith('checkpoint_best')]

        if not checkpoints:
            return None

        latest_checkpoint = max(checkpoints,
                                key=lambda x: int(x.split('_')[1].split('.')[0]))
        return os.path.join(self.checkpoint_path, latest_checkpoint)

    def get_latest_experiment_name(self) -> Optional[str]:
        """Get the name of the latest experiment that has checkpoints"""
        base_dir = os.path.dirname(os.path.dirname(__file__))
        runs_path = os.path.join(base_dir, self.experiment_dir, self.runs_dir)
        
        if not os.path.exists(runs_path):
            return None

        # Get all valid experiment directories
        experiments = [f for f in os.listdir(runs_path)
                      if len(f) == 15 and f[:8].isdigit() and f[8] == '_' and f[9:].isdigit()]

        if not experiments:
            return None

        # Sort experiments by date (newest first)
        experiments.sort(reverse=True)
        
        # Find the latest experiment that has checkpoints
        for exp in experiments:
            checkpoint_dir = os.path.join(runs_path, exp, self.model_save_dir)
            if os.path.exists(checkpoint_dir):
                checkpoints = [f for f in os.listdir(checkpoint_dir)
                             if f.startswith('checkpoint_') and f.endswith('.pth')]
                if checkpoints:
                    return exp
        
        return None

    @property
    def log_path(self) -> str:
        """Get the log path"""
        return os.path.join(self.experiment_path, self.log_dir)
