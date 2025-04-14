# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: SFT trainer
"""

import torch
from tqdm import tqdm
import numpy as np
import os
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from .base_trainer import BaseTrainer


class SFTTrainer(BaseTrainer):
    def __init__(self, model, criterion, optimizer, scheduler, device, save_freq: int = 100):
        super().__init__(model, criterion, optimizer, scheduler, device, save_freq)

    def train_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        num_batches = len(dataloader)
        all_preds = []
        all_labels = []

        progress_bar = tqdm(dataloader, desc='Training')
        for batch_idx, (stock_data, spread_data, labels) in enumerate(progress_bar):
            # Move data to device
            stock_data = stock_data.to(self.device)
            spread_data = spread_data.to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self.model(stock_data, spread_data)
            loss = self.criterion(outputs, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Store predictions and labels for metric calculation
            # Convert back to [-1, 0, 1]
            preds = outputs.argmax(dim=1).cpu().numpy() - 1
            # Convert back to [-1, 0, 1]
            labels_np = labels.cpu().numpy() - 1
            all_preds.extend(preds)
            all_labels.extend(labels_np)

            # Update loss
            total_loss += loss.item()

            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
            })

        # Calculate metrics at epoch end
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        avg_accuracy = accuracy_score(all_labels, all_preds)
        avg_precision = precision_score(all_labels, all_preds, average='macro')
        avg_recall = recall_score(all_labels, all_preds, average='macro')
        avg_f1 = f1_score(all_labels, all_preds, average='macro')

        return total_loss / num_batches, avg_accuracy, avg_precision, avg_recall, avg_f1

    def validate_epoch(self, dataloader):
        self.model.eval()
        total_loss = 0
        num_batches = len(dataloader)
        all_preds = []
        all_labels = []

        with torch.no_grad():
            progress_bar = tqdm(dataloader, desc='Validating')
            for batch_idx, (stock_data, spread_data, labels) in enumerate(progress_bar):
                # Move data to device
                stock_data = stock_data.to(self.device)
                spread_data = spread_data.to(self.device)
                labels = labels.to(self.device)

                # Forward pass
                outputs = self.model(stock_data, spread_data)
                loss = self.criterion(outputs, labels)

                # Store predictions and labels for metric calculation
                # Convert back to [-1, 0, 1]
                preds = outputs.argmax(dim=1).cpu().numpy() - 1
                # Convert back to [-1, 0, 1]
                labels_np = labels.cpu().numpy() - 1
                all_preds.extend(preds)
                all_labels.extend(labels_np)

                # Update loss
                total_loss += loss.item()

                # Update progress bar
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'avg_loss': f'{total_loss / (batch_idx + 1):.4f}'
                })

        # Calculate metrics at epoch end
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        avg_accuracy = accuracy_score(all_labels, all_preds)
        avg_precision = precision_score(all_labels, all_preds, average='macro')
        avg_recall = recall_score(all_labels, all_preds, average='macro')
        avg_f1 = f1_score(all_labels, all_preds, average='macro')

        return total_loss / num_batches, avg_accuracy, avg_precision, avg_recall, avg_f1

    def train(self, train_loader, val_loader, config):
        """Full training loop with early stopping and continue training support
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            config: TrainConfig object containing training parameters
        """
        # Set up directories
        self.model_save_dir = config.checkpoint_path
        os.makedirs(self.model_save_dir, exist_ok=True)

        self.setup_tensorboard(config.log_path)

        start_epoch = 0
        if config.continue_training:
            # If no specific experiment is specified, get the latest one
            if config.continue_from_experiment is None:
                config.continue_from_experiment = config.get_latest_experiment_name()
                if config.continue_from_experiment:
                    self.logger.info(
                        f"Found latest experiment: {config.continue_from_experiment}")
                else:
                    self.logger.info(
                        "No previous experiments found. Starting from scratch.")
                    config.continue_training = False

            if config.continue_training:  # Still True if we found an experiment
                latest_checkpoint = config.get_latest_checkpoint()
                if latest_checkpoint:
                    self.logger.info(
                        f"Loading checkpoint from {latest_checkpoint}")
                    try:
                        checkpoint = torch.load(
                            latest_checkpoint, map_location=self.device)

                        # Load model state
                        self.model.load_state_dict(
                            checkpoint['model_state_dict'])

                        # Load optimizer state
                        self.optimizer.load_state_dict(
                            checkpoint['optimizer_state_dict'])

                        # Load scheduler state
                        self.scheduler.load_state_dict(
                            checkpoint['scheduler_state_dict'])

                        # Load training state
                        start_epoch = checkpoint['epoch'] + 1
                        self.best_loss = checkpoint['best_loss']

                        self.logger.info(
                            f"Successfully loaded checkpoint. Continuing from epoch {start_epoch}")
                        self.logger.info(
                            f"Best validation loss so far: {self.best_loss:.4f}")

                    except Exception as e:
                        self.logger.error(
                            f"Error loading checkpoint: {str(e)}")
                        self.logger.info("Starting training from scratch")
                        start_epoch = 0
                        config.continue_training = False
                else:
                    self.logger.info(
                        f"No checkpoints found in experiment {config.continue_from_experiment}. Starting from scratch.")
                    config.continue_training = False

        for epoch in range(start_epoch, config.num_epochs):
            # Training phase
            train_loss, train_accuracy, train_precision, train_recall, train_f1 = self.train_epoch(
                train_loader)

            # Validation phase
            val_loss, val_accuracy, val_precision, val_recall, val_f1 = self.validate_epoch(
                val_loader)

            # Learning rate scheduling
            self.scheduler.step(val_loss)

            # Logging
            self.logger.info(f'Epoch {epoch+1}/{config.num_epochs}:')
            self.logger.info(f'Training Loss: {train_loss:.4f}')
            self.logger.info(f'Validation Loss: {val_loss:.4f}')

            # Log metrics to TensorBoard
            self.log_metrics(epoch, train_loss, val_loss, train_accuracy, val_accuracy,
                             train_precision, val_precision, train_recall, val_recall,
                             train_f1, val_f1)

            # Save regular checkpoint
            if epoch % self.save_freq == 0:
                self.save_checkpoint(epoch, self.model_save_dir, best=False)

            # Save best model checkpoint
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.save_checkpoint(epoch, self.model_save_dir, best=True)
                self.patience_counter = 0
            else:
                self.patience_counter += 1

            # Early stopping check
            if config.early_stopping and self.patience_counter >= config.patience:
                self.logger.info(
                    f'Early stopping triggered after {epoch+1} epochs')
                break

        self.cleanup()
