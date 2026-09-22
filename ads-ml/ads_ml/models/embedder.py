"""
ADS Embedder - Custom text embedding model.

Generates dense vector representations for semantic search and similarity.
"""

from typing import Optional
from dataclasses import dataclass
from pathlib import Path
import json

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_ml.models.blocks import ADSEncoder


@dataclass
class EmbedderConfig:
    """Configuration for ADSEmbedder."""
    vocab_size: int = 32000
    embed_dim: int = 384
    preset: str = 'small'
    dim: int = 384
    depth: int = 6
    num_heads: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    normalize: bool = True
    pool: str = 'mean'  # 'cls', 'mean', 'max'

    def to_dict(self) -> dict:
        return vars(self)

    @classmethod
    def from_dict(cls, d: dict) -> 'EmbedderConfig':
        return cls(**d)


class ADSEmbedder(nn.Module):
    """
    Custom text embedding model.

    Features:
    - Configurable embedding dimensions
    - Multiple pooling strategies
    - L2 normalization for cosine similarity
    - Contrastive learning support
    - ONNX export ready
    """

    def __init__(self, config: EmbedderConfig):
        super().__init__()
        self.config = config

        # Build encoder
        self.encoder = ADSEncoder.from_preset(
            preset=config.preset,
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
            pool=config.pool,
        )

        # Projection to embedding dim if different
        if config.embed_dim != config.dim:
            self.projection = nn.Linear(config.dim, config.embed_dim)
        else:
            self.projection = nn.Identity()

        self.normalize = config.normalize

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Generate embeddings.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Mask for padding [batch, seq_len]

        Returns:
            Embeddings [batch, embed_dim]
        """
        pooled = self.encoder(input_ids, attention_mask)
        embeddings = self.projection(pooled)

        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=-1)

        return embeddings

    def encode(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Alias for forward, for API compatibility."""
        return self.forward(input_ids, attention_mask)

    def similarity(
        self,
        embeddings1: torch.Tensor,
        embeddings2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute cosine similarity between embedding batches.

        Args:
            embeddings1: [batch1, embed_dim]
            embeddings2: [batch2, embed_dim]

        Returns:
            Similarity matrix [batch1, batch2]
        """
        if self.normalize:
            return torch.matmul(embeddings1, embeddings2.T)
        else:
            return F.cosine_similarity(
                embeddings1.unsqueeze(1),
                embeddings2.unsqueeze(0),
                dim=-1,
            )

    def contrastive_loss(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: Optional[torch.Tensor] = None,
        temperature: float = 0.05,
    ) -> torch.Tensor:
        """
        Compute contrastive loss for training.

        InfoNCE loss with in-batch negatives.
        """
        # Token ids are integer tensors. Comparing the last dimension to
        # embed_dim mis-reads a sequence whose length equals the embedding
        # size as an embedding and skips the encoder.
        if anchor.dtype in (torch.long, torch.int, torch.int32, torch.int64):
            anchor = self.forward(anchor)
            positive = self.forward(positive)
            if negative is not None:
                negative = self.forward(negative)

        batch_size = anchor.shape[0]

        # Similarity with positives
        pos_sim = (anchor * positive).sum(dim=-1) / temperature

        if negative is not None:
            # Explicit negatives
            neg_sim = torch.matmul(anchor, negative.T) / temperature
            logits = torch.cat([pos_sim.unsqueeze(1), neg_sim], dim=1)
        else:
            # In-batch negatives
            all_sim = torch.matmul(anchor, positive.T) / temperature
            # Mask out diagonal (positive pairs)
            mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool)
            all_sim = all_sim.masked_fill(mask, float('-inf'))
            logits = torch.cat([pos_sim.unsqueeze(1), all_sim], dim=1)

        labels = torch.zeros(batch_size, dtype=torch.long, device=anchor.device)
        return F.cross_entropy(logits, labels)

    def save(self, path: Path):
        """Save model and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        with open(path / 'config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        torch.save(self.state_dict(), path / 'model.pt')

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'ADSEmbedder':
        """Load model from directory."""
        path = Path(path)

        with open(path / 'config.json') as f:
            config = EmbedderConfig.from_dict(json.load(f))

        model = cls(config)
        model.load_state_dict(torch.load(path / 'model.pt', map_location=device))
        return model

    def export_onnx(self, path: Path, seq_len: int = 128):
        """Export to ONNX format."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        self.eval()

        dummy_ids = torch.zeros(1, seq_len, dtype=torch.long)
        dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)

        torch.onnx.export(
            self,
            (dummy_ids, dummy_mask),
            str(path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['embeddings'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'seq'},
                'attention_mask': {0: 'batch', 1: 'seq'},
                'embeddings': {0: 'batch'},
            },
            opset_version=17,
        )


def create_embedder(
    embed_dim: int = 384,
    size: str = 'small',
    vocab_size: int = 32000,
    normalize: bool = True,
) -> ADSEmbedder:
    """
    Factory function to create an embedder.

    Args:
        embed_dim: Output embedding dimension
        size: Model size preset
        vocab_size: Vocabulary size
        normalize: Whether to L2 normalize embeddings

    Returns:
        ADSEmbedder instance
    """
    presets = ADSEncoder.CONFIGS
    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(presets.keys())}")

    preset_config = presets[size].copy()
    # Encoder presets say "heads". EmbedderConfig says "num_heads".
    if 'heads' in preset_config:
        preset_config['num_heads'] = preset_config.pop('heads')

    config = EmbedderConfig(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        preset=size,
        normalize=normalize,
        **preset_config,
    )

    return ADSEmbedder(config)
