"""
ADS Custom Model Architectures

Our own models built from scratch - not HuggingFace wrappers.
Optimized for:
- Fast training on small-medium datasets
- Efficient inference (ONNX export ready)
- Go/GoLearn compatible exports
"""

from ads_ml.models.classifier import ADSClassifier
from ads_ml.models.embedder import ADSEmbedder
from ads_ml.models.generator import ADSGenerator
from ads_ml.models.blocks import (
    ADSAttention,
    ADSFeedForward,
    ADSTransformerBlock,
    ADSEncoder,
)

__all__ = [
    "ADSClassifier",
    "ADSEmbedder",
    "ADSGenerator",
    "ADSAttention",
    "ADSFeedForward",
    "ADSTransformerBlock",
    "ADSEncoder",
]
