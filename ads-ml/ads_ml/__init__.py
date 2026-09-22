"""
ADS Model Maker - Custom AI Model Training Toolkit

Build your own models from your data. No ML expertise required.
"""

__version__ = "0.1.0"
__author__ = "Afterdark Solutions"

# Lazy imports to avoid requiring torch for utility scripts
__all__ = [
    "ADSClassifier",
    "ADSEmbedder",
    "ADSGenerator",
    "ADSTrainer",
]


def __getattr__(name):
    """Lazy import to avoid torch dependency for non-ML tools."""
    if name == "ADSClassifier":
        from ads_ml.models.classifier import ADSClassifier
        return ADSClassifier
    elif name == "ADSEmbedder":
        from ads_ml.models.embedder import ADSEmbedder
        return ADSEmbedder
    elif name == "ADSGenerator":
        from ads_ml.models.generator import ADSGenerator
        return ADSGenerator
    elif name == "ADSTrainer":
        from ads_ml.training.trainer import ADSTrainer
        return ADSTrainer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
