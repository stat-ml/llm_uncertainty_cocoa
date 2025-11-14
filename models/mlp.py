import torch
import torch.nn as nn

class MLP(nn.Module):
    """
    Multi-Layer Perceptron model for predicting similarity scores from embeddings.
    
    Architecture:
    1. Projection layers: Input -> 2x head_dim -> head_dim with LayerNorm and GELU
    2. Classifier layers: head_dim -> interim_dim -> 1 (final prediction)
    """
    def __init__(self, input_dim, head_dim, interim_dim, dropout):
        """
        Args:
            input_dim (int): Dimension of input embeddings
            head_dim (int): Dimension of hidden layers
            interim_dim (int): Dimension of intermediate classifier layer
            dropout (float): Dropout probability for regularization
        """
        super().__init__()
        # Projection network: reduces dimensionality while preserving information
        self.proj = nn.Sequential(
            nn.Linear(input_dim, head_dim * 2),  # First expand dimensions
            nn.LayerNorm(head_dim * 2),          # Normalize for stable training
            nn.GELU(),                           # Non-linear activation
            nn.Dropout(dropout),                 # Regularization
            nn.Linear(head_dim * 2, head_dim),   # Then reduce to final projection dim
            nn.LayerNorm(head_dim),
            nn.GELU(),
        )

        # Classifier network: maps projected embeddings to similarity scores
        self.classifier = nn.Sequential(
            nn.Linear(head_dim, interim_dim),    # Project to intermediate space
            nn.LayerNorm(interim_dim),           # Normalize
            nn.GELU(),                           # Non-linear activation
            nn.Dropout(p=dropout),               # Additional regularization
            nn.Linear(interim_dim, 1)            # Final prediction layer
        )

    def forward(self, embeddings, labels=None):
        """
        Forward pass through the network.

        Args:
            embeddings (torch.Tensor): Input embeddings
            labels (torch.Tensor, optional): Ground truth labels for loss computation

        Returns:
            dict: Contains model outputs and optionally the loss if labels provided
        """
        embeddings = embeddings.to(torch.float32)  # Ensure float32 precision
        x = self.proj(embeddings)                  # Project embeddings
        x = self.classifier(x)                     # Generate predictions

        outputs = {"logits": x}

        if labels is not None:
            labels = labels.view_as(x)             # Reshape labels to match predictions
            loss_fn = nn.MSELoss()                # Mean squared error for regression
            loss = loss_fn(x, labels)
            outputs["loss"] = loss

        return outputs
