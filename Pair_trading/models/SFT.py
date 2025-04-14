import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer models"""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # Apply sine to even indices
        pe[:, 0::2] = torch.sin(position * div_term)
        # Apply cosine to odd indices
        pe[:, 1::2] = torch.cos(position * div_term)

        # Add batch dimension and store
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch_size, seq_length, d_model)
        """
        return self.dropout(x + self.pe[:, :x.size(1)])


class StockEncoder(nn.Module):
    """Encoder for individual stock price sequences"""

    def __init__(self, feature_dim, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.feature_proj = nn.Linear(feature_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, seq_length, features)
        x = self.feature_proj(x)
        x = self.pos_encoder(x)
        return self.transformer(x)


class StockFusion(nn.Module):
    """Fusion module for multiple stock representations"""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, num_stocks, seq_length, d_model)
        batch_size, num_stocks, seq_length, d_model = x.shape

        # Reshape for transformer: (batch_size * seq_length, num_stocks, d_model)
        x = x.transpose(1, 2).reshape(
            batch_size * seq_length, num_stocks, d_model)

        # Process
        x = self.transformer(x)

        # Reshape back: (batch_size, seq_length, d_model)
        x = x.mean(dim=1).reshape(batch_size, seq_length, d_model)
        return x


class SpreadEncoder(nn.Module):
    """Encoder for spread sequences"""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.value_proj = nn.Linear(1, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers)

    def forward(self, x):
        # x shape: (batch_size, seq_length, 1)
        x = self.value_proj(x)
        x = self.pos_encoder(x)
        return self.transformer(x)


class FinalFusion(nn.Module):
    """Final fusion between stock and spread representations"""

    def __init__(self, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()

        # Cross attention for spread attending to stock representations
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Self attention for final fusion
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )

        # Final MLP for processing the fused representation
        self.fusion_mlp = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.LayerNorm(dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.LayerNorm(d_model)
        )

    def forward(self, stock_repr, spread_repr):
        # Cross attention: spread attending to stock
        fused_repr, _ = self.cross_attention(
            query=spread_repr,
            key=stock_repr,
            value=stock_repr
        )

        # Combine fused representation with spread representation
        # Concatenate along sequence dimension
        combined = torch.cat([fused_repr, spread_repr], dim=1)

        # Self attention to fuse the combined representation
        final_repr, _ = self.self_attention(
            query=combined,
            key=combined,
            value=combined
        )

        # Global pooling
        final_pooled = final_repr.mean(dim=1)  # (batch_size, d_model)

        # Final processing
        return self.fusion_mlp(final_pooled)


class StockFusionTransformer(nn.Module):
    def __init__(
        self,
        max_stocks: int = 5,
        feature_dim: int = 5,  # OHLCV features
        d_model: int = 64,
        nhead: int = 4,
        num_encoder_layers: int = 2,
        dim_feedforward: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Individual stock encoders
        self.stock_encoders = nn.ModuleList([
            StockEncoder(
                feature_dim=feature_dim,
                d_model=d_model,
                nhead=nhead,
                num_layers=num_encoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            ) for _ in range(max_stocks)
        ])

        # Stock fusion module
        self.stock_fusion = StockFusion(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Spread encoder
        self.spread_encoder = SpreadEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Final fusion module
        self.final_fusion = FinalFusion(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, 3)  # 3 classes: -1, 0, 1
        )

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights using different strategies for different components"""

        def _init_transformer_weights(module):
            if isinstance(module, nn.Linear):
                # Xavier uniform initialization for linear layers
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.LayerNorm):
                # LayerNorm initialization
                nn.init.constant_(module.weight, 1.0)
                nn.init.constant_(module.bias, 0)

            elif isinstance(module, nn.MultiheadAttention):
                # Initialize attention weights
                if hasattr(module, 'in_proj_weight') and module.in_proj_weight is not None:
                    nn.init.xavier_uniform_(module.in_proj_weight)
                if hasattr(module, 'out_proj.weight'):
                    nn.init.xavier_uniform_(module.out_proj.weight)
                if hasattr(module, 'in_proj_bias') and module.in_proj_bias is not None:
                    nn.init.constant_(module.in_proj_bias, 0.)
                if hasattr(module, 'out_proj.bias'):
                    nn.init.constant_(module.out_proj.bias, 0.)

        def _init_classifier_weights(module):
            if isinstance(module, nn.Linear):
                # Kaiming initialization for classifier layers
                nn.init.kaiming_normal_(
                    module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

        # Initialize transformer components
        for encoder in self.stock_encoders:
            encoder.apply(_init_transformer_weights)
        self.stock_fusion.apply(_init_transformer_weights)
        self.spread_encoder.apply(_init_transformer_weights)
        self.final_fusion.apply(_init_transformer_weights)

        # Initialize classifier separately
        self.classifier.apply(_init_classifier_weights)

    def _reset_parameters(self):
        """Reset parameters using Xavier initialization"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def init_from_pretrained(self, pretrained_model_path: str):
        """Initialize from a pretrained model"""
        try:
            pretrained_dict = torch.load(
                pretrained_model_path, map_location='cpu')
            model_dict = self.state_dict()

            # Filter out unnecessary keys
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and v.shape == model_dict[k].shape}

            # Update model state dict
            model_dict.update(pretrained_dict)
            self.load_state_dict(model_dict)

            print(
                f"Successfully loaded pretrained weights from {pretrained_model_path}")
            print(f"Loaded {len(pretrained_dict)}/{len(model_dict)} layers")

        except Exception as e:
            print(f"Error loading pretrained weights: {str(e)}")
            self._initialize_weights()  # Fallback to standard initialization

    def forward(self, x: torch.Tensor, spread: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Stock data tensor (batch_size, num_stocks, seq_length, features)
            spread: Spread tensor (batch_size, seq_length)
        Returns:
            Trading signals: (batch_size, 3)
        """
        batch_size, num_stocks, seq_length, features = x.shape

        # 1. Encode individual stocks
        stock_encodings = []
        for i in range(num_stocks):
            stock_data = x[:, i]  # (batch_size, seq_length, features)
            # (batch_size, seq_length, d_model)
            encoded = self.stock_encoders[i](stock_data)
            stock_encodings.append(encoded)

        # Stack stock encodings
        # (batch_size, num_stocks, seq_length, d_model)
        stock_encodings = torch.stack(stock_encodings, dim=1)

        # 2. Fuse stock information
        # (batch_size, seq_length, d_model)
        fused_stocks = self.stock_fusion(stock_encodings)

        # 3. Process spread
        spread = spread.unsqueeze(-1)  # (batch_size, seq_length, 1)
        # (batch_size, seq_length, d_model)
        spread_repr = self.spread_encoder(spread)

        # 4. Final fusion between stocks and spread
        final_repr = self.final_fusion(
            fused_stocks, spread_repr)  # (batch_size, d_model)

        # 5. Classification
        return self.classifier(final_repr)


# Example usage
if __name__ == "__main__":
    # Model initialization
    model = StockFusionTransformer(
        max_stocks=5,
        feature_dim=5,
        d_model=64,
        nhead=4
    )

    # Example batches
    batch_size = 32
    seq_length = 10
    features = 5

    # Test with different number of stocks
    x1 = torch.randn(batch_size, 2, seq_length, features)  # 2 stocks
    spread1 = torch.randn(batch_size, seq_length)
    out1 = model(x1, spread1)
    print(f"Output shape for 2 stocks: {out1.shape}")

    x2 = torch.randn(batch_size, 3, seq_length, features)  # 3 stocks
    spread2 = torch.randn(batch_size, seq_length)
    out2 = model(x2, spread2)
    print(f"Output shape for 3 stocks: {out2.shape}")
