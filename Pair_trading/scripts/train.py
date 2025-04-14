# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Training script
"""

import os
import torch
import pandas as pd
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from config.data_config import DataConfig
from config.train_config import TrainConfig
from config.model_config import ModelConfig
from utils.dataset import Dataset_1min, DataLoader_1min
from trainers.SFT_trainer import SFTTrainer
from models.SFT import StockFusionTransformer

# def setup_directories(config):
#     """Setup necessary directories for the experiment"""
#     base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#     experiment_dir = config.experiment_dir
#     experiment_dir = os.path.join(base_dir, experiment_dir)
#     os.makedirs(experiment_dir, exist_ok=True)

#     runs_dir = os.path.join(experiment_dir, config.runs_dir)
#     os.makedirs(runs_dir, exist_ok=True)

#     specific_experiment_dir = os.path.join(runs_dir, config.experiment_name)
#     os.makedirs(specific_experiment_dir, exist_ok=True)

#     save_dir = os.path.join(specific_experiment_dir, config.model_save_dir)
#     log_dir = os.path.join(specific_experiment_dir, config.log_dir)

#     os.makedirs(save_dir, exist_ok=True)
#     os.makedirs(log_dir, exist_ok=True)

#     return save_dir, log_dir


def read_data(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = config.data_dir
    processed_data_dir = config.processed_data_dir
    interval = config.interval
    train_begin_year = config.train_begin_year
    train_end_year = config.train_end_year
    val_begin_year = config.val_begin_year
    val_end_year = config.val_end_year
    try:
        train_data_path = os.path.join(base_dir, data_dir, processed_data_dir, f"Aligned_{interval}_{train_begin_year}_{train_end_year}_data.csv")
        val_data_path = os.path.join(base_dir, data_dir, processed_data_dir, f"Aligned_{interval}_{val_begin_year}_{val_end_year}_data.csv")
        train_data = pd.read_csv(train_data_path)
        val_data = pd.read_csv(val_data_path)
        return train_data, val_data

    except Exception as e:
        print(f"Error reading data: {str(e)}")

def main():
    # Load configurations
    model_config = ModelConfig()
    train_config = TrainConfig()
    data_config = DataConfig()

    # Setup directories
    # save_dir, log_dir = setup_directories(train_config)
    train_df, val_df = read_data(data_config)

    # Initialize model
    model = StockFusionTransformer(
        max_stocks=model_config.max_stocks,
        feature_dim=model_config.feature_dim,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_encoder_layers=model_config.num_encoder_layers,
        dim_feedforward=model_config.dim_feedforward,
        dropout=model_config.dropout
    )

    # Initialize dataset and dataloaders
    train_dataset = Dataset_1min(
        df = train_df,
        pairs=data_config.pairs,
        seq_length=data_config.seq_length,
        pred_length=data_config.pred_length,
        ema_window_size=data_config.ema_window_size,
        hedge_ratios=data_config.hedge_ratios
    )

    val_dataset = Dataset_1min(
        df = val_df,
        pairs=data_config.pairs,
        seq_length=data_config.seq_length,
        pred_length=data_config.pred_length,
        ema_window_size=data_config.ema_window_size,
        hedge_ratios=data_config.hedge_ratios
    )

    train_loader = DataLoader_1min(
        train_dataset,
        batch_size=train_config.batch_size,
        shuffle=True,
        drop_last=True
    )

    val_loader = DataLoader_1min(
        val_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False
    )

    # Initialize training components
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(
        model.parameters(),
        lr=train_config.learning_rate,
        betas=(train_config.beta1, train_config.beta2),
        eps=train_config.eps,
        weight_decay=train_config.weight_decay
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=train_config.scheduler_factor,
        patience=train_config.scheduler_patience,
        min_lr=train_config.min_lr
    )

    # Initialize trainer
    trainer = SFTTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=train_config.device,
        save_freq=train_config.save_freq
    )

    # Start training
    trainer.train(train_loader, val_loader, train_config)


if __name__ == "__main__":
    main()
