# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Base trainer
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import os
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class BaseTrainer(ABC):
    def __init__(self, model, criterion, optimizer, scheduler, device, save_freq: int = 100):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.save_freq = save_freq

        self.best_loss = float('inf')
        self.patience_counter = 0
        self.writer = None

        # Setup logging
        self.logger = self._setup_logger()

        self.model.to(self.device)

    def _setup_logger(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)

        # Create handlers
        c_handler = logging.StreamHandler()
        c_handler.setLevel(logging.INFO)

        # Create formatters and add it to handlers
        c_format = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        c_handler.setFormatter(c_format)

        # Add handlers to the logger
        logger.addHandler(c_handler)

        return logger

    def setup_tensorboard(self, log_dir: str) -> None:
        """Setup TensorBoard writer"""
        os.makedirs(log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir)

    def save_checkpoint(self, epoch: int, save_dir: str, best: bool = False) -> None:
        """Save model checkpoint"""
        os.makedirs(save_dir, exist_ok=True)
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_loss': self.best_loss
        }
        if best:
            file_name = "checkpoint_best.pth"
        else:
            file_name = f"checkpoint_{epoch}.pth"
        path = os.path.join(save_dir, file_name)
        torch.save(checkpoint, path)
        self.logger.info(f"Checkpoint saved: {path}")

    def load_checkpoint(self, path: str) -> Dict[str, Any]:
        """Load model checkpoint"""
        self.logger.info(f"Loading checkpoint: {path}")
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.best_loss = checkpoint['best_loss']
        return checkpoint

    @abstractmethod
    def train_epoch(self, dataloader):
        """Train for one epoch"""
        pass

    @abstractmethod
    def validate_epoch(self, dataloader):
        """Validate for one epoch"""
        pass

    def should_stop_early(self, epoch: int, val_loss: float, patience: int) -> bool:
        """Check if training should stop early"""
        if val_loss < self.best_loss:
            self.best_loss = val_loss
            self.patience_counter = 0
            self.save_checkpoint(epoch, self.model_save_dir, best=True)
            return False

        self.patience_counter += 1
        return self.patience_counter >= patience

    def log_metrics(self, epoch: int, train_loss: float, val_loss: float, train_accuracy: float, val_accuracy: float, train_precision: float, val_precision: float, train_recall: float, val_recall: float, train_f1: float, val_f1: float) -> None:
        """Log metrics to TensorBoard"""
        if self.writer is not None:
            self.writer.add_scalar('Loss/train', train_loss, epoch)
            self.writer.add_scalar('Loss/validation', val_loss, epoch)
            self.writer.add_scalar(
                'Learning_rate', self.optimizer.param_groups[0]['lr'], epoch)
            self.writer.add_scalar(
                "Train/Accuracy", train_accuracy, epoch
            )
            self.writer.add_scalar(
                "Train/Precision", train_precision, epoch
            )
            self.writer.add_scalar(
                "Train/Recall", train_recall, epoch
            )
            self.writer.add_scalar(
                "Train/F1", train_f1, epoch
            )
            self.writer.add_scalar(
                "Validation/Accuracy", val_accuracy, epoch
            )
            self.writer.add_scalar(
                "Validation/Precision", val_precision, epoch
            )
            self.writer.add_scalar(
                "Validation/Recall", val_recall, epoch
            )
            self.writer.add_scalar(
                "Validation/F1", val_f1, epoch
            )

    def cleanup(self) -> None:
        """Cleanup resources"""
        if self.writer is not None:
            self.writer.close()
