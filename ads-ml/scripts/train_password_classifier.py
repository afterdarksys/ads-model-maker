#!/usr/bin/env python3
"""
Train a password strength classifier on RockYou2024 data.

Usage:
    python train_password_classifier.py --bucket ml-datasets --prefix rockyou2024-strong/
    python train_password_classifier.py --local /path/to/passwords.txt --max-passwords 1000000

The model learns to classify passwords into:
- WEAK: Crackable in seconds to minutes (common patterns, dictionary words)
- MEDIUM: Crackable in hours to days (some patterns but more complex)
- STRONG: Would require significant compute resources
"""

import argparse
import os
import sys
from pathlib import Path
from datetime import datetime
import random

import torch
from torch.utils.data import DataLoader, random_split
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from ads_ml.models.classifier import create_classifier, ADSClassifier
from ads_ml.datasets.password import (
    PasswordDataset,
    PasswordLabeler,
    PasswordStrengthTier,
    StreamingPasswordDataset,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train password strength classifier")

    # Data source
    data_group = parser.add_mutually_exclusive_group(required=True)
    data_group.add_argument("--bucket", help="OCI bucket name")
    data_group.add_argument("--local", help="Local file or directory path")

    parser.add_argument("--prefix", default="", help="OCI object prefix")
    parser.add_argument("--max-passwords", type=int, default=5_000_000,
                       help="Maximum passwords to load")

    # Model configuration
    parser.add_argument("--model-size", default="small",
                       choices=["micro", "tiny", "small", "base", "medium", "large"])
    parser.add_argument("--max-length", type=int, default=64,
                       help="Maximum password length")

    # Training configuration
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-steps", type=int, default=500)

    # Labeling configuration
    parser.add_argument("--weak-threshold", type=float, default=35.0,
                       help="Score threshold below which passwords are WEAK")
    parser.add_argument("--strong-threshold", type=float, default=60.0,
                       help="Score threshold above which passwords are STRONG")

    # Output
    parser.add_argument("--output-dir", type=Path, default=Path("./output/password-classifier"))
    parser.add_argument("--save-every", type=int, default=1000,
                       help="Save checkpoint every N steps")

    # Hardware
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num-workers", type=int, default=4)

    return parser.parse_args()


def load_dataset(args) -> PasswordDataset:
    """Load dataset from OCI or local file."""
    labeler = PasswordLabeler(
        weak_threshold=args.weak_threshold,
        strong_threshold=args.strong_threshold,
    )

    print(f"Loading passwords (max {args.max_passwords:,})...")

    if args.bucket:
        dataset = PasswordDataset.from_oci(
            bucket=args.bucket,
            prefix=args.prefix,
            labeler=labeler,
            max_passwords=args.max_passwords,
            max_length=args.max_length,
        )
    else:
        local_path = Path(args.local)
        if local_path.is_dir():
            # Load all .txt files in directory
            passwords = []
            for txt_file in local_path.glob("**/*.txt"):
                if not txt_file.is_file():
                    continue
                if len(passwords) >= args.max_passwords:
                    break
                with open(txt_file, "r", encoding="utf-8", errors="replace") as f:
                    for line in f:
                        if len(passwords) >= args.max_passwords:
                            break
                        pw = line.strip()
                        if pw:
                            passwords.append(pw)
            dataset = PasswordDataset(passwords, labeler=labeler, max_length=args.max_length)
        else:
            dataset = PasswordDataset.from_file(
                local_path,
                labeler=labeler,
                max_passwords=args.max_passwords,
                max_length=args.max_length,
            )

    print(f"Loaded {len(dataset):,} passwords")

    # Print label distribution
    label_counts = {}
    for label in dataset.labels:
        label_counts[label.value] = label_counts.get(label.value, 0) + 1
    print("Label distribution:")
    for label, count in sorted(label_counts.items()):
        pct = count / len(dataset) * 100
        print(f"  {label}: {count:,} ({pct:.1f}%)")

    return dataset


def create_dataloaders(dataset: PasswordDataset, args):
    """Split dataset and create dataloaders."""
    # 90/10 train/val split
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    print(f"Train: {len(train_dataset):,}, Val: {len(val_dataset):,}")

    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, scheduler, device, epoch, args):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for step, batch in enumerate(train_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        output = model(input_ids, attention_mask, labels)
        loss = output["loss"]
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        predictions = output["predictions"]
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

        if (step + 1) % 100 == 0:
            avg_loss = total_loss / (step + 1)
            accuracy = correct / total * 100
            lr = scheduler.get_last_lr()[0]
            print(f"  Step {step + 1}/{len(train_loader)} | Loss: {avg_loss:.4f} | "
                  f"Acc: {accuracy:.2f}% | LR: {lr:.2e}")

        # Save checkpoint
        if args.save_every and (step + 1) % args.save_every == 0:
            checkpoint_path = args.output_dir / f"checkpoint-epoch{epoch}-step{step + 1}"
            model.save(checkpoint_path)
            print(f"  Saved checkpoint: {checkpoint_path}")

    return total_loss / len(train_loader), correct / total


def evaluate(model, val_loader, device):
    """Evaluate on validation set."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    class_correct = {}
    class_total = {}

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            output = model(input_ids, attention_mask, labels)
            total_loss += output["loss"].item()

            predictions = output["predictions"]
            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            # Per-class accuracy
            for pred, label in zip(predictions, labels):
                label_val = label.item()
                class_total[label_val] = class_total.get(label_val, 0) + 1
                if pred == label:
                    class_correct[label_val] = class_correct.get(label_val, 0) + 1

    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total

    # Print per-class accuracy
    tier_names = [t.value for t in PasswordStrengthTier]
    print("  Per-class accuracy:")
    for label_id, total_count in sorted(class_total.items()):
        correct_count = class_correct.get(label_id, 0)
        acc = correct_count / total_count * 100
        label_name = tier_names[label_id] if label_id < len(tier_names) else f"label_{label_id}"
        print(f"    {label_name}: {acc:.2f}% ({correct_count}/{total_count})")

    return avg_loss, accuracy


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Password Strength Classifier Training")
    print("=" * 60)
    print(f"Device: {args.device}")
    print(f"Model size: {args.model_size}")
    print(f"Output: {args.output_dir}")
    print()

    # Load data
    dataset = load_dataset(args)
    train_loader, val_loader = create_dataloaders(dataset, args)

    # Create model
    model = create_classifier(
        num_labels=len(PasswordStrengthTier),
        label_names=[t.value for t in PasswordStrengthTier],
        size=args.model_size,
        vocab_size=128,  # ASCII characters
        problem_type="single_label",
    )
    model = model.to(args.device)

    print(f"\nModel: {model.get_num_params():,} parameters ({model.get_model_size_mb():.2f} MB)")

    # Optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    total_steps = len(train_loader) * args.epochs
    scheduler = CosineAnnealingLR(optimizer, T_max=total_steps)

    # Training loop
    best_val_acc = 0
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")
        print("-" * 40)

        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, args.device, epoch, args
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc * 100:.2f}%")

        val_loss, val_acc = evaluate(model, val_loader, args.device)
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc * 100:.2f}%")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_path = args.output_dir / "best"
            model.save(best_path)
            print(f"  New best model saved! ({val_acc * 100:.2f}%)")

    # Save final model
    final_path = args.output_dir / "final"
    model.save(final_path)
    print(f"\nFinal model saved to: {final_path}")

    # Export ONNX
    onnx_path = args.output_dir / "model.onnx"
    model.export_onnx(onnx_path, seq_len=args.max_length)
    print(f"ONNX exported to: {onnx_path}")

    # Summary
    print("\n" + "=" * 60)
    print("Training Complete!")
    print(f"Best validation accuracy: {best_val_acc * 100:.2f}%")
    print(f"Model saved to: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
