"""
ADS Custom Neural Network Building Blocks

Efficient, ONNX-exportable components for building custom models.
Designed for fast training and inference on domain-specific data.
"""

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ADSAttention(nn.Module):
    """
    Efficient multi-head attention with optional flash attention.

    Features:
    - Rotary position embeddings (RoPE)
    - Optional causal masking
    - ONNX export compatible
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        dropout: float = 0.0,
        causal: bool = False,
        use_rope: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim or dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.causal = causal
        self.use_rope = use_rope

        inner_dim = self.head_dim * num_heads

        self.qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.out_proj = nn.Linear(inner_dim, dim, bias=False)
        self.dropout = nn.Dropout(dropout)

        if use_rope:
            self.rotary = RotaryEmbedding(self.head_dim)

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x)
        q, k, v = rearrange(
            qkv, 'b n (three h d) -> three b h n d',
            three=3, h=self.num_heads
        )

        # Apply rotary embeddings
        if self.use_rope:
            q = self.rotary(q)
            k = self.rotary(k)

        # Attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Causal mask
        if self.causal:
            causal_mask = torch.triu(
                torch.ones(N, N, device=x.device, dtype=torch.bool),
                diagonal=1
            )
            attn = attn.masked_fill(causal_mask, float('-inf'))

        # Optional attention mask (convert to bool if needed)
        if mask is not None:
            mask_bool = mask.bool() if mask.dtype != torch.bool else mask
            attn = attn.masked_fill(~mask_bool.unsqueeze(1).unsqueeze(2), float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        return self.out_proj(out)


class RotaryEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) for better position awareness."""

    def __init__(self, dim: int, max_seq_len: int = 8192, base: int = 10000):
        super().__init__()
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        self.max_seq_len = max_seq_len
        self._build_cache(max_seq_len)

    def _build_cache(self, seq_len: int):
        t = torch.arange(seq_len, device=self.inv_freq.device)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat([freqs, freqs], dim=-1)
        self.register_buffer('cos_cached', emb.cos()[None, None, :, :])
        self.register_buffer('sin_cached', emb.sin()[None, None, :, :])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        seq_len = x.shape[-2]
        if seq_len > self.max_seq_len:
            self._build_cache(seq_len)

        cos = self.cos_cached[:, :, :seq_len, :]
        sin = self.sin_cached[:, :, :seq_len, :]

        return (x * cos) + (self._rotate_half(x) * sin)

    @staticmethod
    def _rotate_half(x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)


class ADSFeedForward(nn.Module):
    """
    SwiGLU Feed-Forward Network for better gradient flow.

    More efficient than standard FFN for similar parameter count.
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__()
        hidden_dim = hidden_dim or int(dim * 8 / 3)
        # Round to multiple of 64 for efficiency
        hidden_dim = ((hidden_dim + 63) // 64) * 64

        self.w1 = nn.Linear(dim, hidden_dim, bias=bias)
        self.w2 = nn.Linear(hidden_dim, dim, bias=bias)
        self.w3 = nn.Linear(dim, hidden_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # SwiGLU activation
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - faster than LayerNorm."""

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x * norm).type_as(x) * self.weight


class ADSTransformerBlock(nn.Module):
    """
    Single transformer block with pre-norm architecture.

    Components:
    - RMSNorm (faster than LayerNorm)
    - Multi-head attention with RoPE
    - SwiGLU feed-forward
    """

    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        head_dim: Optional[int] = None,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        causal: bool = False,
    ):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = ADSAttention(
            dim=dim,
            num_heads=num_heads,
            head_dim=head_dim,
            dropout=dropout,
            causal=causal,
        )

        self.norm2 = RMSNorm(dim)
        self.ff = ADSFeedForward(
            dim=dim,
            hidden_dim=int(dim * ff_mult),
            dropout=dropout,
        )

    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), mask=mask)
        x = x + self.ff(self.norm2(x))
        return x


class ADSEncoder(nn.Module):
    """
    Full encoder stack for classification/embedding tasks.

    Configurable depth and width for different model sizes.
    """

    # Preset configurations
    CONFIGS = {
        'micro': {'dim': 128, 'depth': 2, 'heads': 2},
        'tiny': {'dim': 256, 'depth': 4, 'heads': 4},
        'small': {'dim': 384, 'depth': 6, 'heads': 6},
        'base': {'dim': 512, 'depth': 8, 'heads': 8},
        'medium': {'dim': 768, 'depth': 12, 'heads': 12},
        'large': {'dim': 1024, 'depth': 16, 'heads': 16},
    }

    def __init__(
        self,
        vocab_size: int,
        dim: int = 384,
        depth: int = 6,
        num_heads: int = 6,
        max_seq_len: int = 512,
        dropout: float = 0.1,
        pool: str = 'cls',  # 'cls', 'mean', 'max'
    ):
        super().__init__()
        self.dim = dim
        self.pool = pool

        # Token embeddings
        self.embed = nn.Embedding(vocab_size, dim)
        self.embed_dropout = nn.Dropout(dropout)

        # CLS token for pooling
        if pool == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            ADSTransformerBlock(
                dim=dim,
                num_heads=num_heads,
                dropout=dropout,
                causal=False,
            )
            for _ in range(depth)
        ])

        self.norm = RMSNorm(dim)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_preset(
        cls,
        preset: str,
        vocab_size: int,
        **kwargs,
    ) -> 'ADSEncoder':
        """Create encoder from preset configuration."""
        config = cls.CONFIGS[preset].copy()
        # Remap 'heads' to 'num_heads' for consistency
        if 'heads' in config:
            config['num_heads'] = config.pop('heads')
        config.update(kwargs)
        return cls(vocab_size=vocab_size, **config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        B, N = input_ids.shape

        # Embed tokens
        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        # Prepend CLS token
        if self.pool == 'cls':
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
            if attention_mask is not None:
                attention_mask = F.pad(attention_mask, (1, 0), value=True)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask=attention_mask)

        x = self.norm(x)

        # Pool to single vector
        if self.pool == 'cls':
            return x[:, 0]
        elif self.pool == 'mean':
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                return (x * mask).sum(1) / mask.sum(1)
            return x.mean(dim=1)
        elif self.pool == 'max':
            return x.max(dim=1).values

        return x  # Return all tokens

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_config(self) -> dict:
        """Get model configuration for saving."""
        return {
            'dim': self.dim,
            'depth': len(self.blocks),
            'num_heads': self.blocks[0].attn.num_heads,
            'vocab_size': self.embed.num_embeddings,
            'pool': self.pool,
        }
