# Portions of this file are adapted from LeWorldModel.
# Copyright (c) 2026 Lucas Maes
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Causal action-conditioned Transformer predictor used by LeWM.

The architecture is adapted from the official LeWorldModel implementation:
https://github.com/lucas-maes/le-wm/blob/main/module.py (MIT License).

State embeddings form the Transformer token sequence.  At every time step, an
already-encoded action embedding modulates both the attention and feed-forward
branches through AdaLN-Zero.  Causal self-attention ensures that the prediction
at time ``t`` can only depend on state/action pairs through time ``t``.
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


class Mish(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x * torch.tanh(nn.functional.softplus(x))


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int, scale: float = 1.0):
        super().__init__()
        self.dim = dim
        self.scale = scale

    def forward(self, x: Tensor) -> Tensor:
        x = x.float() * self.scale
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000.0) / max(half_dim - 1, 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[..., None] * emb
        return torch.cat((emb.sin(), emb.cos()), dim=-1)
    
def modulate(x: Tensor, shift: Tensor, scale: Tensor) -> Tensor:
    """Apply adaptive LayerNorm shift and scale parameters."""
    return x * (1 + scale) + shift


class FeedForward(nn.Module):
    """Pre-normalized Transformer feed-forward network."""

    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class Attention(nn.Module):
    """Multi-head scaled dot-product attention with causal masking by default."""

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        # Kept for state/API parity with the reference implementation. PyTorch's
        # scaled_dot_product_attention applies this scale internally.
        self.scale = dim_head**-0.5
        self.dropout = dropout
        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

    def forward(self, x: Tensor, causal: bool = True) -> Tensor:
        """Attend over a sequence shaped ``[batch, time, dim]``."""
        x = self.norm(x)
        dropout_p = self.dropout if self.training else 0.0
        query, key, value = self.to_qkv(x).chunk(3, dim=-1)
        query, key, value = (
            rearrange(tensor, "b t (h d) -> b h t d", h=self.heads)
            for tensor in (query, key, value)
        )
        output = F.scaled_dot_product_attention(
            query,
            key,
            value,
            dropout_p=dropout_p,
            is_causal=causal,
        )
        output = rearrange(output, "b h t d -> b t (h d)")
        return self.to_out(output)


class ConditionalBlock(nn.Module):
    """Causal Transformer block with per-token AdaLN-Zero conditioning."""

    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.mlp = FeedForward(dim, mlp_dim, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.norm2 = nn.LayerNorm(dim, elementwise_affine=False, eps=1e-6)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(dim, 6 * dim, bias=True),
        )

        # The residual branches initially contribute zero, then learn how each
        # action shifts, scales, and gates attention and MLP features.
        nn.init.constant_(self.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.adaLN_modulation[-1].bias, 0)

    def forward(self, x: Tensor, c: Tensor) -> Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            shift_mlp,
            scale_mlp,
            gate_mlp,
        ) = self.adaLN_modulation(c).chunk(6, dim=-1)

        x = x + gate_msa * self.attn(modulate(self.norm1(x), shift_msa, scale_msa))
        x = x + gate_mlp * self.mlp(modulate(self.norm2(x), shift_mlp, scale_mlp))
        return x


class Transformer(nn.Module):
    """Causal Transformer with per-token AdaLN-Zero conditioning."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_dim)
        self.layers = nn.ModuleList([])

        self.input_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.cond_proj = (
            nn.Linear(input_dim, hidden_dim)
            if input_dim != hidden_dim
            else nn.Identity()
        )
        self.output_proj = (
            nn.Linear(hidden_dim, output_dim)
            if hidden_dim != output_dim
            else nn.Identity()
        )

        for _ in range(depth):
            self.layers.append(
                ConditionalBlock(hidden_dim, heads, dim_head, mlp_dim, dropout)
            )

    def forward(
        self,
        x: Tensor,
        c: Tensor | None = None,
        flow_condition: Tensor | None = None,
        bottleneck_condition: Tensor | None = None,
    ) -> Tensor:
        x = self.input_proj(x)
        if c is not None:
            c = self.cond_proj(c)
            if flow_condition is not None:
                c = c + flow_condition

        for index, block in enumerate(self.layers):
            if bottleneck_condition is not None and index == len(self.layers) // 2:
                x = x + bottleneck_condition
            if c is None:
                raise ValueError("condition is required by a conditional block.")
            x = block(x, c)

        return self.output_proj(self.norm(x))
