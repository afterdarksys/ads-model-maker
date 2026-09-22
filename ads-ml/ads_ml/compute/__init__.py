"""
ADS Compute Module - GPU cloud provider integrations.

Supports:
- Vast.ai for cheap GPU training
- OCI for persistent infrastructure
"""

__all__ = ["VastAIProvider", "GPUTier"]


def __getattr__(name):
    """Lazy import."""
    if name == "VastAIProvider":
        from ads_ml.compute.vastai_provider import VastAIProvider
        return VastAIProvider
    elif name == "GPUTier":
        from ads_ml.compute.vastai_provider import GPUTier
        return GPUTier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
