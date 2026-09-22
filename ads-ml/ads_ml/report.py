"""Model and dataset footprint: parameters, disk, and a RAM estimate.

The RAM number is an upper bound for activations, not a measured allocation.
It answers the question the training form has to ask before a run starts:
how big is this model, and how much data is going into it.
"""

from __future__ import annotations


def model_footprint(
    *,
    num_params: int,
    dtype_bytes: int = 4,
    seq_len: int = 512,
    batch_size: int = 1,
    hidden_dim: int = 0,
    depth: int = 0,
    data_chars: int = 0,
    data_docs: int = 0,
) -> dict:
    if num_params < 0:
        raise ValueError("num_params cannot be negative")
    if dtype_bytes < 1:
        raise ValueError("dtype_bytes must be at least 1")

    disk_bytes = num_params * dtype_bytes
    activation_bytes = 0
    if hidden_dim > 0 and depth > 0 and seq_len > 0 and batch_size > 0:
        # Weights stay resident. Activations are roughly one hidden state
        # per layer, plus attention scores (seq by seq) at the same width.
        hidden_state = batch_size * seq_len * hidden_dim * dtype_bytes
        attention_scores = batch_size * seq_len * seq_len * dtype_bytes
        activation_bytes = depth * (hidden_state + attention_scores)

    return {
        "parameters": num_params,
        "dtype_bytes": dtype_bytes,
        "disk_bytes": disk_bytes,
        "disk_mb": disk_bytes / (1024 * 1024),
        "estimated_activation_ram_bytes": activation_bytes,
        "estimated_peak_ram_bytes": disk_bytes + activation_bytes,
        "estimated_peak_ram_mb": (disk_bytes + activation_bytes) / (1024 * 1024),
        "data_docs": data_docs,
        "data_chars": data_chars,
    }


def footprint_from_module(model, **kwargs) -> dict:
    """Read parameter count off a module that implements get_num_params."""
    if hasattr(model, "get_num_params"):
        num_params = int(model.get_num_params())
    else:
        num_params = sum(int(param.numel()) for param in model.parameters())

    config = getattr(model, "config", None)
    if config is not None:
        kwargs.setdefault("hidden_dim", int(getattr(config, "dim", 0) or 0))
        kwargs.setdefault("depth", int(getattr(config, "depth", 0) or 0))
        kwargs.setdefault("seq_len", int(getattr(config, "max_seq_len", 512) or 512))
    return model_footprint(num_params=num_params, **kwargs)
