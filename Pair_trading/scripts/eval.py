
import os
import torch
import logging
from tqdm import tqdm
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.SFT import StockFusionTransformer
from trainers.SFT_trainer import SFTTrainer
from utils.dataset import Dataset_1min, DataLoader_1min
from config.model_config import ModelConfig
from config.train_config import TrainConfig
from config.data_config import DataConfig


def read_data(config):
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = config.data_dir
    processed_data_dir = config.processed_data_dir
    interval = config.interval
    test_begin_year = config.test_begin_year
    test_end_year = config.test_end_year

    try:
        test_data_path = os.path.join(base_dir, data_dir, processed_data_dir,
                                       f"Aligned_{interval}_{test_begin_year}_{test_end_year}_data.csv")
        test_data = pd.read_csv(test_data_path)
        return test_data

    except Exception as e:
        print(f"Error reading data: {str(e)}")

def setup_logger():
    logger = logging.getLogger('evaluation')
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger


def load_model_checkpoint(model, config, logger):
    """Load either best or latest checkpoint based on configuration"""
    if config.continue_from_experiment is None:
        config.continue_from_experiment = config.get_latest_experiment_name()
        if not config.continue_from_experiment:
            raise FileNotFoundError("No experiments found")
        logger.info(
            f"Using latest experiment: {config.continue_from_experiment}")

    # Try to load best checkpoint first
    best_model_path = os.path.join(
        config.checkpoint_path, 'checkpoint_best.pth')
    if os.path.exists(best_model_path):
        logger.info(f"Loading best checkpoint from {best_model_path}")
        checkpoint = torch.load(best_model_path, map_location=config.device)
    else:
        # Fall back to latest checkpoint
        latest_checkpoint = config.get_latest_checkpoint()
        if not latest_checkpoint:
            raise FileNotFoundError("No checkpoints found")
        logger.info(f"Loading latest checkpoint from {latest_checkpoint}")
        checkpoint = torch.load(latest_checkpoint, map_location=config.device)

    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    return model, checkpoint['epoch']


def plot_confusion_matrix(cm, save_path):
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(save_path)
    plt.close()


def evaluate(model, test_loader, device, logger):
    model.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch_idx, (stock_data, spread_data, labels) in enumerate(tqdm(test_loader, desc='Evaluating')):
            stock_data = stock_data.to(device)
            spread_data = spread_data.to(device)
            labels = labels.to(device)

            outputs = model(stock_data, spread_data)
            # Convert back to [-1, 0, 1]
            preds = outputs.argmax(dim=1).cpu().numpy() - 1
            labels_np = labels.cpu().numpy() - 1  # Convert back to [-1, 0, 1]

            all_preds.extend(preds)
            all_labels.extend(labels_np)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    report = classification_report(all_labels, all_preds,
                                   target_names=['Short (-1)', 'Hold (0)', 'Long (1)'])
    cm = confusion_matrix(all_labels, all_preds)

    return report, cm, all_preds, all_labels


def main():
    # Load configurations
    train_config = TrainConfig()
    data_config = DataConfig()
    model_config = ModelConfig()

    # Setup logger
    logger = setup_logger()

    test_df = read_data(data_config)

    # Create model
    model = StockFusionTransformer(
        max_stocks=model_config.max_stocks,
        feature_dim=model_config.feature_dim,
        d_model=model_config.d_model,
        nhead=model_config.nhead,
        num_encoder_layers=model_config.num_encoder_layers,
        dim_feedforward=model_config.dim_feedforward,
        dropout=model_config.dropout
    )

    # Load model checkpoint
    model, epoch = load_model_checkpoint(model, train_config, logger)
    model.to(train_config.device)

    # Load test dataset
    # Note: This part needs to be updated based on your train.py implementation
    # Placeholder for dataset loading code:
    test_dataset = Dataset_1min(
        df=test_df,
        seq_length=data_config.seq_length,
        pred_length=data_config.pred_length,
        pairs=data_config.pairs,
        ema_window_size=data_config.ema_window_size,
        hedge_ratios=data_config.hedge_ratios
    )

    test_loader = DataLoader_1min(
        dataset=test_dataset,
        batch_size=train_config.batch_size,
        shuffle=False,
        drop_last=False
    )

    # Evaluate
    report, cm, predictions, labels = evaluate(
        model, test_loader, train_config.device, logger)

    # Save results
    results_dir = os.path.join(
        train_config.experiment_path, 'evaluation_results')
    os.makedirs(results_dir, exist_ok=True)

    # Save classification report
    with open(os.path.join(results_dir, 'classification_report.txt'), 'w') as f:
        f.write(f"Evaluation results for epoch {epoch}\n\n")
        f.write(report)

    # Plot and save confusion matrix
    plot_confusion_matrix(cm, os.path.join(
        results_dir, 'confusion_matrix.png'))

    # Save predictions
    np.save(os.path.join(results_dir, 'predictions.npy'), predictions)
    np.save(os.path.join(results_dir, 'true_labels.npy'), labels)

    logger.info(f"\nClassification Report:\n{report}")
    logger.info(
        "Evaluation completed. Results saved in evaluation_results directory.")


if __name__ == "__main__":
    main()
