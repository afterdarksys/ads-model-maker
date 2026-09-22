#!/usr/bin/env python3
"""
Export trained model to ONNX and create quantized version.

Usage:
    python export_onnx.py --model-dir ./output/gpu-trained --output-dir ./output/onnx
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from ads_ml.models.classifier import ADSClassifier


def export_onnx(model_dir: Path, output_dir: Path, seq_len: int = 64):
    """Export model to ONNX format."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading model from {model_dir}...")
    model = ADSClassifier.load(model_dir, device='cpu')

    print(f"Model size: {model.get_model_size_mb():.1f} MB")
    print(f"Parameters: {model.get_num_params():,}")

    # Export to ONNX
    onnx_path = output_dir / "model.onnx"
    print(f"\nExporting to ONNX: {onnx_path}")
    model.export_onnx(onnx_path, seq_len=seq_len)

    onnx_size = onnx_path.stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {onnx_size:.1f} MB")

    return onnx_path


def quantize_onnx(onnx_path: Path, output_dir: Path):
    """Quantize ONNX model for smaller size and faster inference."""
    try:
        from onnxruntime.quantization import quantize_dynamic, QuantType
    except ImportError:
        print("\nonnxruntime not installed, skipping quantization")
        print("Install with: pip install onnxruntime")
        return None

    quant_path = output_dir / "model_quantized.onnx"
    print(f"\nQuantizing to: {quant_path}")

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(quant_path),
        weight_type=QuantType.QUInt8,
    )

    quant_size = quant_path.stat().st_size / (1024 * 1024)
    print(f"Quantized model size: {quant_size:.1f} MB")

    return quant_path


def verify_onnx(onnx_path: Path, model_dir: Path):
    """Verify ONNX model produces same outputs as PyTorch."""
    try:
        import onnxruntime as ort
        import numpy as np
    except ImportError:
        print("\nonnxruntime not installed, skipping verification")
        return

    print(f"\nVerifying ONNX model...")

    # Load PyTorch model
    model = ADSClassifier.load(model_dir, device='cpu')
    model.eval()

    # Create test input
    test_input = torch.randint(0, 256, (1, 64), dtype=torch.long)
    test_mask = torch.ones(1, 64, dtype=torch.bool)

    # PyTorch inference
    with torch.no_grad():
        pt_output = model(test_input, test_mask)
        pt_logits = pt_output['logits'].numpy()

    # ONNX inference
    session = ort.InferenceSession(str(onnx_path))
    onnx_logits = session.run(
        ['logits'],
        {
            'input_ids': test_input.numpy(),
            'attention_mask': test_mask.numpy(),
        }
    )[0]

    # Compare
    diff = np.abs(pt_logits - onnx_logits).max()
    print(f"Max difference between PyTorch and ONNX: {diff:.6f}")

    if diff < 1e-4:
        print("✓ ONNX model verified!")
    else:
        print("⚠ Warning: Outputs differ significantly")


def create_go_config(output_dir: Path, model_dir: Path, seq_len: int):
    """Create Go-compatible config file."""
    import json

    # Load labels
    with open(model_dir / 'labels.json') as f:
        labels = json.load(f)

    # Load model config
    with open(model_dir / 'config.json') as f:
        config = json.load(f)

    go_config = {
        'model_file': 'model_quantized.onnx',
        'labels': labels,
        'seq_len': seq_len,
        'vocab_size': config.get('vocab_size', 256),
        'num_labels': len(labels),
    }

    config_path = output_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(go_config, f, indent=2)

    print(f"\nGo config saved to: {config_path}")
    return config_path


def main():
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--model-dir", type=Path, required=True,
                       help="Directory containing trained model")
    parser.add_argument("--output-dir", type=Path, required=True,
                       help="Output directory for ONNX files")
    parser.add_argument("--seq-len", type=int, default=64,
                       help="Sequence length for export (default: 64)")
    parser.add_argument("--skip-quantize", action="store_true",
                       help="Skip quantization step")
    parser.add_argument("--skip-verify", action="store_true",
                       help="Skip verification step")

    args = parser.parse_args()

    # Export to ONNX
    onnx_path = export_onnx(args.model_dir, args.output_dir, args.seq_len)

    # Quantize
    if not args.skip_quantize:
        quant_path = quantize_onnx(onnx_path, args.output_dir)

    # Verify
    if not args.skip_verify:
        verify_onnx(onnx_path, args.model_dir)

    # Create Go config
    create_go_config(args.output_dir, args.model_dir, args.seq_len)

    print("\n" + "=" * 50)
    print("Export complete!")
    print("=" * 50)
    print(f"\nFiles created in {args.output_dir}:")
    for f in sorted(args.output_dir.iterdir()):
        size = f.stat().st_size
        if size > 1024 * 1024:
            size_str = f"{size / (1024*1024):.1f} MB"
        elif size > 1024:
            size_str = f"{size / 1024:.1f} KB"
        else:
            size_str = f"{size} B"
        print(f"  {f.name}: {size_str}")


if __name__ == "__main__":
    main()
