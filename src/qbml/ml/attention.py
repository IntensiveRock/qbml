import torch
from torch import nn
from torch.nn import functional as F


class MultiheadAttention(nn.Module):
    """
    A class to wrap many AttentionHead()'s.

    This class will hand multiple attention heads in parallel and concatenate results.
    """

    def __init__(
        self,
        num_heads: int,
        n_embd: int,
        block_size: int = None,  # length of the sim
        tgt_size: int = None,
        mask: bool = False,
    ):
        """Initialize the MultiheadAttention class object."""
        super().__init__()
        head_size = n_embd // num_heads
        self.heads = nn.ModuleList(
            [
                AttentionHead(
                    n_embd,
                    head_size,
                    block_size=block_size,
                    tgt_size=tgt_size,
                    mask=mask,
                )
                for head in range(num_heads)
            ]
        )
        # self.Wo = nn.Linear(n_embd, n_embd)
        self.Wo = WeightMatrix(block_size * num_heads, n_embd)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor = None,
        v: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """Define the forward pass for multihead attention."""
        concat_attn = torch.cat(
            [head(q, k, v, tgt_padding_mask) for head in self.heads], dim=-1
        )
        output = self.Wo(concat_attn)
        return output


class AttentionHead(nn.Module):
    """
    An attention mechanism.

    As described in attention is all you need.
    """

    def __init__(
        self,
        n_embd: int,
        head_size: int,
        block_size: int = None,
        tgt_size: int = None,
        mask: bool = False,
    ):
        """Initialize the attention head object."""
        super().__init__()
        self.Qw = WeightMatrix(n_embd, block_size)
        self.Kw = WeightMatrix(n_embd, block_size)
        self.Vw = WeightMatrix(n_embd, block_size)
        if mask:
            self.register_buffer("tril", torch.tril(torch.ones(tgt_size, tgt_size)))
            self.mask = mask
        else:
            self.mask = mask

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor = None,
        value: torch.Tensor = None,
        tgt_padding_mask: torch.Tensor = None,
    ) -> torch.Tensor:
        """
        Define the forward pass for the attention head.

        Q, K, V matricies are assumed the same and are the function argument.
        The first operation is the result of the dot product between x,
        and their respective weight matricies.
        """

        q = self.Qw(query)
        k = self.Kw(key)
        v = self.Vw(value)
        A, B, C = q.shape
        qkT = q @ k.transpose(-2, -1) / C**-0.5
        if self.mask:
            qkT = qkT.masked_fill(self.tril[:B, :B] == 0, float("-inf"))
        if tgt_padding_mask is not None:
            tgt_padding_mask = tgt_padding_mask.unsqueeze(1)
            # tgt_padding_mask = tgt_padding_mask.unsqueeze(1)
            # print("The shape of mask after second unsqueeze: ", tgt_padding_mask)
            # qkT = qkT + tgt_padding_mask
            # print("This is qkT after padding: ", qkT)
        attn_scores = F.softmax(qkT, dim=-1)
        attn_scaled_values = attn_scores @ v
        return attn_scaled_values


class WeightMatrix(nn.Module):
    """
    A transformer weight matrix.
    """
    
    def __init__(self,
                 features_in,
                 features_out
                 ):
        super().__init__()
        self.weight = nn.Parameter(
            torch.rand(features_in, features_out, dtype=torch.float),
            requires_grad=True
        )


    def forward(self,
                x,
                ):
        return x @ self.weight
