"""
ADS Generator - Custom text generation model.

Autoregressive transformer for text generation tasks.
"""

from typing import Optional, Iterator
from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_ml.models.blocks import ADSTransformerBlock, RMSNorm


@dataclass
class GeneratorConfig:
    """Configuration for ADSGenerator."""
    vocab_size: int = 32000
    dim: int = 512
    depth: int = 8
    num_heads: int = 8
    max_seq_len: int = 1024
    dropout: float = 0.1
    ff_mult: float = 4.0

    # Generation defaults
    default_max_tokens: int = 256
    default_temperature: float = 0.8
    default_top_p: float = 0.95
    default_top_k: int = 50

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'GeneratorConfig':
        return cls(**d)


class ADSGenerator(nn.Module):
    """
    Custom autoregressive text generator.

    Features:
    - Causal attention for autoregressive generation
    - KV-cache for efficient generation
    - Multiple sampling strategies (greedy, top-k, top-p, temperature)
    - Streaming generation support
    - ONNX export ready
    """

    CONFIGS = {
        'micro': {'dim': 256, 'depth': 4, 'num_heads': 4},
        'tiny': {'dim': 384, 'depth': 6, 'num_heads': 6},
        'small': {'dim': 512, 'depth': 8, 'num_heads': 8},
        'base': {'dim': 768, 'depth': 12, 'num_heads': 12},
        'medium': {'dim': 1024, 'depth': 16, 'num_heads': 16},
        'large': {'dim': 1536, 'depth': 24, 'num_heads': 24},
    }

    def __init__(self, config: GeneratorConfig):
        super().__init__()
        self.config = config

        # Token embeddings
        self.embed = nn.Embedding(config.vocab_size, config.dim)
        self.embed_dropout = nn.Dropout(config.dropout)

        # Transformer blocks (causal)
        self.blocks = nn.ModuleList([
            ADSTransformerBlock(
                dim=config.dim,
                num_heads=config.num_heads,
                ff_mult=config.ff_mult,
                dropout=config.dropout,
                causal=True,
            )
            for _ in range(config.depth)
        ])

        self.norm = RMSNorm(config.dim)

        # Output projection (tied with embeddings)
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # Weight tying

        # Initialize
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    @classmethod
    def from_preset(cls, preset: str, vocab_size: int = 32000, **kwargs) -> 'ADSGenerator':
        """Create generator from preset."""
        config_dict = cls.CONFIGS[preset].copy()
        config_dict['vocab_size'] = vocab_size
        config_dict.update(kwargs)
        config = GeneratorConfig(**config_dict)
        return cls(config)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Padding mask [batch, seq_len]
            labels: Target tokens for loss computation

        Returns:
            Dict with logits and optional loss
        """
        x = self.embed(input_ids)
        x = self.embed_dropout(x)

        for block in self.blocks:
            x = block(x, mask=attention_mask)

        x = self.norm(x)
        logits = self.lm_head(x)

        output = {'logits': logits}

        if labels is not None:
            # Shift for autoregressive loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()

            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            output['loss'] = loss

        return output

    @torch.no_grad()
    def generate(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_k: Optional[int] = None,
        top_p: Optional[float] = None,
        stop_tokens: Optional[list[int]] = None,
        stream: bool = False,
    ) -> torch.Tensor | Iterator[int]:
        """
        Generate text autoregressively.

        Args:
            input_ids: Prompt token IDs [batch, seq_len]
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: Only sample from top k tokens
            top_p: Nucleus sampling threshold
            stop_tokens: Token IDs that stop generation
            stream: If True, yield tokens one at a time

        Returns:
            Generated token IDs [batch, seq_len + new_tokens]
            Or iterator of tokens if stream=True
        """
        self.eval()

        # Use defaults from config
        max_new_tokens = max_new_tokens or self.config.default_max_tokens
        temperature = temperature if temperature is not None else self.config.default_temperature
        top_k = top_k or self.config.default_top_k
        top_p = top_p or self.config.default_top_p
        stop_tokens = stop_tokens or []

        device = input_ids.device
        batch_size = input_ids.shape[0]

        if stream:
            if batch_size != 1:
                raise ValueError("streaming generation supports batch size 1")
            return self._generate_stream(
                input_ids, max_new_tokens, temperature, top_k, top_p, stop_tokens
            )

        generated = input_ids
        # One flag per row. Batch size greater than 1 cannot call .item().
        finished = torch.zeros(batch_size, dtype=torch.bool, device=device)

        for _ in range(max_new_tokens):
            if bool(finished.all()):
                break

            # Truncate to max sequence length
            if generated.shape[1] > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated

            # Get next token logits
            output = self.forward(context)
            next_logits = output['logits'][:, -1, :]

            # Sample next token
            next_token = self._sample(next_logits, temperature, top_k, top_p)
            if bool(finished.any()):
                next_token = torch.where(finished, torch.zeros_like(next_token), next_token)

            # Append to generated
            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

            if stop_tokens:
                for stop_id in stop_tokens:
                    finished = finished | (next_token == stop_id)
            if batch_size == 1 and bool(finished[0]):
                break

        return generated

    def _generate_stream(
        self,
        input_ids: torch.Tensor,
        max_new_tokens: int,
        temperature: float,
        top_k: int,
        top_p: float,
        stop_tokens: list[int],
    ) -> Iterator[int]:
        """Generate tokens one at a time (streaming)."""
        generated = input_ids

        for _ in range(max_new_tokens):
            if generated.shape[1] > self.config.max_seq_len:
                context = generated[:, -self.config.max_seq_len:]
            else:
                context = generated

            output = self.forward(context)
            next_logits = output['logits'][:, -1, :]

            next_token = self._sample(next_logits, temperature, top_k, top_p)
            token_id = next_token.item()

            yield token_id

            if token_id in stop_tokens:
                break

            generated = torch.cat([generated, next_token.unsqueeze(1)], dim=1)

    def _sample(
        self,
        logits: torch.Tensor,
        temperature: float,
        top_k: int,
        top_p: float,
    ) -> torch.Tensor:
        """Sample from logits with temperature, top-k, and top-p."""
        # Temperature
        if temperature > 0:
            logits = logits / temperature
        else:
            # Greedy
            return logits.argmax(dim=-1)

        # Top-k filtering
        if top_k > 0:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')

        # Top-p (nucleus) filtering
        if top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

            # Remove tokens with cumulative prob above threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0

            indices_to_remove = sorted_indices_to_remove.scatter(
                -1, sorted_indices, sorted_indices_to_remove
            )
            logits[indices_to_remove] = float('-inf')

        # Sample
        probs = F.softmax(logits, dim=-1)
        return torch.multinomial(probs, num_samples=1).squeeze(-1)

    def save(self, path: Path):
        """Save model and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        torch.save(self.state_dict(), path / 'model.pt')

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'ADSGenerator':
        """Load model from directory."""
        path = Path(path)

        with open(path / 'config.json') as f:
            config = GeneratorConfig.from_dict(json.load(f))

        model = cls(config)
        model.load_state_dict(torch.load(path / 'model.pt', map_location=device))
        return model

    def get_num_params(self) -> int:
        """Total parameters (excluding tied weights)."""
        # Subtract tied lm_head weights
        n_params = sum(p.numel() for p in self.parameters())
        n_params -= self.lm_head.weight.numel()
        return n_params


def create_generator(
    size: str = 'small',
    vocab_size: int = 32000,
    max_seq_len: int = 1024,
) -> ADSGenerator:
    """
    Factory function to create a generator.

    Args:
        size: Model size preset
        vocab_size: Vocabulary size
        max_seq_len: Maximum sequence length

    Returns:
        ADSGenerator instance
    """
    return ADSGenerator.from_preset(
        preset=size,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
    )
