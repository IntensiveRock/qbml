import os
from pathlib import Path

import numpy as np
import torch
from torch import nn
from tqdm import tqdm

from qbml.ml.attention import MultiheadAttention
from qbml.ml.feedforward import FeedForward


class Transformer(nn.Module):

    def __init__(
        self,
        n_encoders: int,
        n_embd: int,
        n_heads: int,
        src_len: int,
        n_tgt: int,
        pred_len: int = None,
        rectifier: str = None,
        hidden_dim: int = False,
        device: str = "cpu",
        block_size: int = False,
    ):
        """
        Construct a QubitML Transformer with specific hyperparameters as described below.

        :param n_encoders:
        :type n_encoders:
        :param src_len:
        :type src_len:
        :param n_tgt:
        :type n_tgt:
        :param pred_len:
        :type pred_len:
        :param hidden_dim:
        :type hidden_dim:
        :param device:
        :type device:
        :param model_name:
        :type model_name:

        """
        super().__init__()
        self.src_len = src_len
        self.n_tgt = n_tgt
        self.pred_len = pred_len
        self.n_embd = n_embd
        self.num_heads = n_heads
        if not block_size:
            self.block_size = self.n_embd
        else:
            self.block_size = block_size
        self.hidden_dim = hidden_dim
        self.device = device
        self.enc_pos_emb = nn.Embedding(src_len, n_embd)
        self.encoder_stack = nn.ModuleList(
            [EncoderBlock(n_embd, self.num_heads, self.block_size) for _ in range(n_encoders)]
        )
        self.ff_pred_len = FeedForward(src_len, 0.0, pred_len)
        self.ff_sembd_tembd = FeedForward(n_embd, 0.0, n_tgt)
        if rectifier == 'gelu':
            self.rectifier = torch.nn.GELU()
        elif rectifier == 'relu':
            self.rectifier = torch.nn.ReLU()
        else:
            self.rectifier = False


    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode part of the forward pass."""
        x = x + self.enc_pos_emb(torch.arange(x.size(-2), device=self.device))
        for block in self.encoder_stack:
            x = block(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Define the forward pass for the NN."""
        enc_output = self.encode(x)
        change_embd = self.ff_sembd_tembd(enc_output)
        change_embd_T = change_embd.transpose(-2, -1)
        logits = self.ff_pred_len(change_embd_T)
        if self.rectifier:
            logits = self.rectifier(logits)
        return logits.transpose(-2, -1)

    def train_loop(self, dataloader, loss_fn, optimizer):
        """Define the training loop for the neural network."""
        self.train()
        training_loss = 0
        for batch, (src, tgt) in enumerate(dataloader):
            src = src.to(self.device)
            tgt = tgt.to(self.device)
            logits = self(src)
            loss = loss_fn(logits, tgt)
            training_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
        return training_loss / len(dataloader)

    def val_loop(self, dataloader, loss_fn):
        """Define the validation loop for the neural network."""
        self.eval()
        val_loss = 0
        for batch, (src, tgt) in enumerate(dataloader):
            src = src.to(device=self.device)
            tgt = tgt.to(device=self.device)
            logits = self(src)
            loss = loss_fn(logits, tgt)
            val_loss += loss.item()
        return val_loss / len(dataloader)


    def predict(self, dataloader: torch.utils.data.DataLoader) -> torch.Tensor:
        """Make predictions with the trained network."""
        self.eval()
        batch_logits = torch.zeros(
            len(dataloader), dataloader.batch_size, self.pred_len, self.n_tgt
        )
        # batch_logits = batch_logits.to(device=self.device)
        for batch, (src, tgt) in tqdm(enumerate(dataloader)):
            src = src.to(device=self.device)
            logits = self(src)
            batch_logits[batch] += logits.to(device='cpu')
        return batch_logits


class EncoderBlock(nn.Module):
    """
    A transformer encoder block.

    This block contains all elements of the encoder.
    """

    def __init__(self, n_embd: int,
                 num_heads: int,
                 block_size: int,
                 dropout: float = 0.0):
        """Initialize the encoder block."""
        super().__init__()
        self.mha = MultiheadAttention(num_heads, n_embd, block_size)
        self.layer_norm_1 = nn.LayerNorm(n_embd)
        self.feed_forward = FeedForward(n_embd, dropout)
        self.layer_norm_2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        """Define the forward pass for the encoder block."""
        x = self.layer_norm_1(x + self.mha(x, x, x))
        out = self.layer_norm_2(x + self.feed_forward(x))
        return out


def trunc(val, decs):
    """Truncate values."""
    tmp = np.trunc(val * 10**decs)
    return tmp / 10**decs
