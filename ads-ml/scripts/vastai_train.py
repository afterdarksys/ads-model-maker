#!/usr/bin/env python3
"""
Run training jobs on Vast.ai GPU instances.

Usage:
    # List available GPUs (default: STANDARD tier)
    python vastai_train.py list --tier budget

    # Estimate cost for a training run
    python vastai_train.py estimate --hours 2 --tier standard

    # Run password classifier training on Vast.ai
    python vastai_train.py run password-classifier \
        --data-url "https://..." \
        --epochs 10 \
        --tier standard

    # Check running instances
    python vastai_train.py instances

    # Destroy an instance
    python vastai_train.py destroy <instance_id>
"""

import argparse
import os
import sys
import json
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from ads_ml.compute.vastai_provider import VastAIProvider, GPUTier, InstanceConfig


def cmd_list(args):
    """List available GPU offers."""
    tier = GPUTier(args.tier)
    provider = VastAIProvider(api_key=args.api_key, tier=tier)

    print(f"\nSearching for GPUs (tier: {tier.value}, max: ${provider.get_max_price():.2f}/hr)...")
    print("-" * 80)

    offers = provider.find_offers(
        min_vram=args.min_vram,
        sort_by=args.sort,
        limit=args.limit,
    )

    if not offers:
        print("No offers found matching criteria.")
        return

    print(f"{'ID':<10} {'GPU':<20} {'VRAM':<8} {'$/hr':<10} {'Value':<10} {'BW':<8}")
    print("-" * 80)

    for offer in offers:
        gpu_name = offer.get("gpu_name", "Unknown")[:19]
        vram = offer.get("gpu_ram", 0) // 1024  # Convert to GB
        price = offer.get("dph_total", 0)
        value = offer.get("_value_score", 0)
        bandwidth = offer.get("inet_down", 0)

        print(f"{offer['id']:<10} {gpu_name:<20} {vram:<8} ${price:<9.3f} {value:<10.1f} {bandwidth:<8.0f}")

    print(f"\nFound {len(offers)} matching offers.")
    print(f"Account balance: ${provider.get_account_balance():.2f}")


def cmd_estimate(args):
    """Estimate training cost."""
    tier = GPUTier(args.tier)
    provider = VastAIProvider(api_key=args.api_key, tier=tier)

    offer = provider.find_best_offer(min_vram=args.min_vram)

    if not offer:
        print("No offers found matching criteria.")
        return

    estimate = provider.estimate_cost(
        price_per_hour=offer["dph_total"],
        hours=args.hours,
    )

    print(f"\n{'='*50}")
    print(f"Cost Estimate: {offer['gpu_name']}")
    print(f"{'='*50}")
    print(f"GPU:           {offer['gpu_name']}")
    print(f"VRAM:          {offer.get('gpu_ram', 0) // 1024} GB")
    print(f"Price/hour:    ${estimate['price_per_hour']:.3f}")
    print(f"Est. hours:    {estimate['estimated_hours']:.1f}")
    print(f"Total cost:    ${estimate['total_cost']:.2f}")
    print(f"{'='*50}")
    print(f"Balance:       ${estimate['current_balance']:.2f}")
    print(f"After run:     ${estimate['remaining_after']:.2f}")
    print(f"Can afford:    {'Yes' if estimate['can_afford'] else 'NO - Insufficient funds!'}")


def cmd_instances(args):
    """List running instances."""
    provider = VastAIProvider(api_key=args.api_key)
    instances = provider.list_instances()

    if not instances:
        print("No running instances.")
        return

    print(f"\n{'ID':<10} {'GPU':<20} {'Status':<12} {'$/hr':<8} {'SSH':<30}")
    print("-" * 80)

    for inst in instances:
        ssh_info = f"{inst.ssh_host}:{inst.ssh_port}" if inst.ssh_host else "N/A"
        print(f"{inst.instance_id:<10} {inst.gpu_name[:19]:<20} {inst.status:<12} ${inst.price_per_hour:<7.3f} {ssh_info:<30}")


def cmd_destroy(args):
    """Destroy an instance."""
    provider = VastAIProvider(api_key=args.api_key)

    if args.all:
        instances = provider.list_instances()
        for inst in instances:
            print(f"Destroying instance {inst.instance_id}...")
            provider.destroy_instance(inst.instance_id)
        print(f"Destroyed {len(instances)} instances.")
    else:
        print(f"Destroying instance {args.instance_id}...")
        provider.destroy_instance(args.instance_id)
        print("Done.")


def cmd_run(args):
    """Run a training job on Vast.ai."""
    tier = GPUTier(args.tier)
    provider = VastAIProvider(api_key=args.api_key, tier=tier)

    # Find best offer
    print(f"\nFinding best GPU (tier: {tier.value}, vram>={args.min_vram}GB)...")
    offer = provider.find_best_offer(min_vram=args.min_vram)

    if not offer:
        print("No offers found matching criteria.")
        sys.exit(1)

    gpu_name = offer["gpu_name"]
    price = offer["dph_total"]
    print(f"Selected: {gpu_name} @ ${price:.3f}/hr")

    # Estimate cost
    est = provider.estimate_cost(price, args.estimated_hours)
    if not est["can_afford"]:
        print(f"Insufficient balance (${est['current_balance']:.2f}) for estimated ${est['total_cost']:.2f}")
        if not args.force:
            sys.exit(1)
        print("--force specified, continuing anyway...")

    # Build training command based on job type
    if args.job == "password-classifier":
        train_cmd = build_password_classifier_cmd(args)
    else:
        print(f"Unknown job type: {args.job}")
        sys.exit(1)

    # Create setup script
    setup_script = """#!/bin/bash
set -e
pip install torch einops
mkdir -p /workspace/output
mkdir -p /workspace/data
"""

    if args.data_url:
        setup_script += f"""
echo "Downloading data..."
cd /workspace/data
wget -q "{args.data_url}" -O data.tar.gz || curl -sL "{args.data_url}" -o data.tar.gz
tar -xzf data.tar.gz || true
ls -la
"""

    # Create instance
    config = InstanceConfig(
        image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        disk_gb=args.disk,
        onstart_script=setup_script,
    )

    print(f"\nCreating instance...")
    instance = provider.create_instance(offer["id"], config)
    print(f"Instance ID: {instance.instance_id}")

    print(f"\nWaiting for instance to be ready...")
    try:
        instance = provider.wait_for_ready(instance.instance_id, timeout=args.timeout)
    except TimeoutError:
        print("Instance failed to start in time. Destroying...")
        provider.destroy_instance(instance.instance_id)
        sys.exit(1)

    print(f"Instance ready: ssh -p {instance.ssh_port} root@{instance.ssh_host}")

    # Copy training code if provided
    if args.code_dir:
        print(f"\nCopying training code from {args.code_dir}...")
        provider.copy_to_instance(instance.instance_id, args.code_dir, "/workspace/code")

    # Run training
    print(f"\nStarting training...")
    print(f"Command: {train_cmd}")

    returncode, stdout, stderr = provider.run_command(
        instance.instance_id,
        train_cmd,
        timeout=3600 * 24,  # 24 hour max
    )

    print("\n" + "=" * 60)
    print("TRAINING OUTPUT")
    print("=" * 60)
    print(stdout)
    if stderr:
        print("\nSTDERR:")
        print(stderr)

    # Download results
    if args.output_local:
        print(f"\nDownloading results to {args.output_local}...")
        Path(args.output_local).mkdir(parents=True, exist_ok=True)
        provider.copy_from_instance(
            instance.instance_id,
            "/workspace/output/",
            args.output_local,
        )

    # Cleanup
    if not args.keep_instance:
        print(f"\nDestroying instance {instance.instance_id}...")
        provider.destroy_instance(instance.instance_id)
    else:
        print(f"\nInstance kept running: ssh -p {instance.ssh_port} root@{instance.ssh_host}")

    print("\nTraining complete!")
    return returncode


def build_password_classifier_cmd(args) -> str:
    """Build command for password classifier training."""
    cmd = [
        "cd /workspace/code &&",
        "python scripts/train_password_classifier.py",
        "--local /workspace/data",
        f"--max-passwords {args.max_passwords}",
        f"--model-size {args.model_size}",
        f"--epochs {args.epochs}",
        f"--batch-size {args.batch_size}",
        "--output-dir /workspace/output",
        "--device cuda",
    ]
    return " ".join(cmd)


def main():
    parser = argparse.ArgumentParser(description="Vast.ai training manager")
    parser.add_argument("--api-key", default=os.environ.get("VAST_API_KEY"),
                       help="Vast.ai API key")

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # List command
    list_p = subparsers.add_parser("list", help="List available GPUs")
    list_p.add_argument("--tier", default="standard", choices=["budget", "standard", "premium"])
    list_p.add_argument("--min-vram", type=int, default=8)
    list_p.add_argument("--sort", default="value", choices=["price", "performance", "value"])
    list_p.add_argument("--limit", type=int, default=15)

    # Estimate command
    est_p = subparsers.add_parser("estimate", help="Estimate training cost")
    est_p.add_argument("--tier", default="standard", choices=["budget", "standard", "premium"])
    est_p.add_argument("--min-vram", type=int, default=12)
    est_p.add_argument("--hours", type=float, default=2.0)

    # Instances command
    inst_p = subparsers.add_parser("instances", help="List running instances")

    # Destroy command
    destroy_p = subparsers.add_parser("destroy", help="Destroy instance(s)")
    destroy_p.add_argument("instance_id", nargs="?", type=int)
    destroy_p.add_argument("--all", action="store_true", help="Destroy all instances")

    # Run command
    run_p = subparsers.add_parser("run", help="Run training job")
    run_p.add_argument("job", choices=["password-classifier"])
    run_p.add_argument("--tier", default="standard", choices=["budget", "standard", "premium"])
    run_p.add_argument("--min-vram", type=int, default=12)
    run_p.add_argument("--disk", type=int, default=50)
    run_p.add_argument("--timeout", type=int, default=300)
    run_p.add_argument("--estimated-hours", type=float, default=2.0)
    run_p.add_argument("--force", action="store_true")
    run_p.add_argument("--keep-instance", action="store_true")

    # Data options
    run_p.add_argument("--data-url", help="URL to download training data")
    run_p.add_argument("--code-dir", help="Local directory with training code to upload")
    run_p.add_argument("--output-local", help="Local directory to download results")

    # Training options
    run_p.add_argument("--max-passwords", type=int, default=500000)
    run_p.add_argument("--model-size", default="small")
    run_p.add_argument("--epochs", type=int, default=5)
    run_p.add_argument("--batch-size", type=int, default=128)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if not args.api_key:
        print("Error: VAST_API_KEY environment variable or --api-key required")
        sys.exit(1)

    commands = {
        "list": cmd_list,
        "estimate": cmd_estimate,
        "instances": cmd_instances,
        "destroy": cmd_destroy,
        "run": cmd_run,
    }

    sys.exit(commands[args.command](args) or 0)


if __name__ == "__main__":
    main()
