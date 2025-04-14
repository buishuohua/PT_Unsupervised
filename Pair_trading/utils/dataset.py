# -*- coding: utf-8 -*-

"""
    @ __Author__ = Yunkai.Gao

    @    Time    : 4/11/25
    @ Description: Dataset
"""

import pandas as pd
import numpy as np
import random
import os
import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, BatchSampler
from typing import Tuple
# Fix the import by using relative import
from .data_process import spread_pairs, ema, create_sequences
from torch.utils.data.sampler import SubsetRandomSampler


class Dataset_1min(Dataset):
    """Dataset class for 1-minute stock data with pairs trading setup"""

    def __init__(self,
                 df: pd.DataFrame,
                 seq_length: int = 3,
                 pred_length: int = 1,
                 pairs: Tuple[Tuple[str, ...], ...] = (),
                 ema_window_size: int = 10,
                 hedge_ratios: Tuple[Tuple[float, ...], ...] = (),
                 seed: int = 42
                 ):
        """
        Initialize dataset for pairs trading.
        
        Args:
            df: DataFrame containing aligned stock data
            seq_length: Length of input sequence (default: 3)
            pred_length: Length of prediction window (default: 2)
            pairs: Tuple of stock groups, each group as a tuple
                   e.g., (('AAPL', 'MSFT'), ('GOOGL', 'META', 'AMZN'))
            ema_window_size: Window size for EMA calculation (default: 10)
            hedge_ratios: Tuple of hedge ratios for each group (default: None, all 1.0)
        """
        self.df = df
        self.seq_length = seq_length
        self.pred_length = pred_length
        self.pairs = pairs
        self.ema_window_size = ema_window_size
        self.hedge_ratios = hedge_ratios
        self.processed_data = {}
        self.seed = seed
        np.random.seed(self.seed)
        random.seed(self.seed)
        torch.manual_seed(self.seed)

        # Process each pair/group
        for pair, hedge_ratio in zip(self.pairs, self.hedge_ratios):
            # Calculate spread and get stock DataFrames
            stocks_df, spread = spread_pairs(self.df, stocks=pair, hedge_ratios=hedge_ratio)

            # Calculate EMA of spread
            spread_ema = ema(spread, window_size=self.ema_window_size)

            # Create sequences
            pair_sequences, spread_sequences, labels = create_sequences(
                stock_dfs=stocks_df,
                spread=spread_ema,
                seq_length=seq_length,
                pred_length=pred_length
            )

            # Store processed data for the pair
            self.processed_data[pair] = {
                'pair_sequences': pair_sequences,
                'spread_sequences': spread_sequences,
                'labels': labels
            }

        # Create index mapping for __getitem__
        self.pair_indices = []
        for pair in self.pairs:
            num_samples = len(self.processed_data[pair]['labels'])
            self.pair_indices.extend([(pair, idx)
                                     for idx in range(num_samples)])

    def __len__(self):
        """Return total number of samples across all pairs"""
        return len(self.pair_indices)

    def __getitem__(self, index):
        """
        Get a single sample.
        
        Returns:
            tuple: (
                pair_data,  # Shape: (num_stocks, seq_length, features)
                spread,     # Shape: (seq_length,)
                label      # Shape: (1,) - Trading signal (-1, 0, or 1)
            )
        """
        pair, idx = self.pair_indices[index]
        pair_data = self.processed_data[pair]

        return (
            # (num_stocks, seq_length, features)
            pair_data['pair_sequences'][idx],
            pair_data['spread_sequences'][idx],  # (seq_length,)
            pair_data['labels'][idx]  # scalar
        )


class DataLoader_1min(DataLoader):
    def __init__(self, dataset, batch_size=256, shuffle=True, drop_last=True):
        # Group indices by number of stocks
        self.pair_size_groups = {}
        for idx in range(len(dataset)):
            pair, _ = dataset.pair_indices[idx]
            num_stocks = len(pair)
            if num_stocks not in self.pair_size_groups:
                self.pair_size_groups[num_stocks] = []
            self.pair_size_groups[num_stocks].append(idx)

        # Create batches for each group size
        self.grouped_batches = self._create_grouped_batches(
            batch_size, shuffle)

        super().__init__(
            dataset,
            batch_size=batch_size,
            sampler=self._create_batch_sampler(),
            drop_last=drop_last,
            collate_fn=self.collate_fn
        )

    def _create_grouped_batches(self, batch_size, shuffle):
        """Create batches for each group size"""
        grouped_batches = []

        for size in sorted(self.pair_size_groups.keys()):
            indices = self.pair_size_groups[size]
            if shuffle:
                indices = np.random.permutation(indices).tolist()

            # Create batches for this group size
            num_samples = len(indices)
            num_batches = num_samples // batch_size

            for i in range(num_batches):
                start_idx = i * batch_size
                batch_indices = indices[start_idx:start_idx + batch_size]
                grouped_batches.append(batch_indices)

        if shuffle:
            np.random.shuffle(grouped_batches)

        return grouped_batches

    def _create_batch_sampler(self):
        """Create a sampler that yields batches of indices"""
        return BatchSampler(
            sampler=SubsetRandomSampler(range(len(self.grouped_batches))),
            batch_size=1,
            drop_last=False
        )

    @staticmethod
    def collate_fn(batch):
        """
        Custom collate function to handle batches.
        """
        # Get the first sample's shape to verify consistency
        # Shape: (num_stocks, seq_length, features)
        first_pair_data = batch[0][0]
        num_stocks, seq_length, num_features = first_pair_data.shape

        # Verify all samples in batch have same dimensions
        for sample in batch:
            assert sample[0].shape == (num_stocks, seq_length, num_features), \
                f"Inconsistent shapes in batch. Expected {(num_stocks, seq_length, num_features)}, got {sample[0].shape}"

        # Unzip the batch
        pair_data, spreads, labels = zip(*batch)

        # Convert to tensors
        # (batch_size, num_stocks, seq_length, features)
        pair_data_tensor = torch.FloatTensor(np.stack(pair_data))
        spreads_tensor = torch.FloatTensor(
            np.stack(spreads))      # (batch_size, seq_length)
        labels_tensor = torch.LongTensor(
            labels)                    # (batch_size,)

        return pair_data_tensor, spreads_tensor, labels_tensor

    def __iter__(self):
        """Custom iterator to yield batches"""
        for batch_indices in self.grouped_batches:
            batch = [self.dataset[idx] for idx in batch_indices]
            yield self.collate_fn(batch)

    def __len__(self):
        """Return the number of batches"""
        return len(self.grouped_batches)


if __name__ == "__main__":
    # Load aligned data
    data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    aligned_data = pd.read_csv(os.path.join(data_dir, "Aligned_1min_data.csv"))
    aligned_data['timestamp'] = pd.to_datetime(
        aligned_data['timestamp'])  # Ensure timestamp is datetime

    # Define pairs as a tuple of tuples with different lengths
    stock_pairs = (
        ('AAPL', 'MSFT'),           # Pair 1
        ('GOOGL', 'META', 'AMD'),  # Triple 1
    )

    # Create dataset with variable-length pairs
    dataset = Dataset_1min(
        df=aligned_data,
        seq_length=10,
        pred_length=5,
        pairs=stock_pairs
    )

    print(f"\nDataset Summary:")
    print(f"Number of pairs/groups: {len(dataset.pairs)}")
    print(f"Total samples: {len(dataset)}")

    # Print samples per pair/group
    for pair in dataset.pairs:
        num_samples = len(dataset.processed_data[pair]['labels'])
        print(f"\nGroup {pair}:")
        print(f"Number of stocks: {len(pair)}")
        print(f"Number of samples: {num_samples}")

    # Create dataloader
    batch_size = 256
    dataloader = DataLoader_1min(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True
    )

    print("\nDataloader Summary:")
    print(f"Batch size: {batch_size}")
    for group_size, indices in dataloader.pair_size_groups.items():
        num_samples = len(indices)
        num_batches = num_samples // batch_size
        print(f"\nGroup size {group_size} stocks:")
        print(f"Total samples: {num_samples}")
        print(f"Number of batches: {num_batches}")
