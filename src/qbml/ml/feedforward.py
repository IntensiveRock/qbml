import torch
from torch import nn


class FeedForward(nn.Module):
    """A simple feedforward network with non-linearity."""

    def __init__(self, n_embd, dropout, n_tgt: int = None):
        """Initialize the feedforward network."""
        super().__init__()
        if n_tgt is None:
            n_tgt = n_embd
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_tgt),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """Define the forward pass for the feedforward network."""
        return self.net(x)
