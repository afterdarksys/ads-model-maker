"""
Vast.ai GPU Provider - Manages GPU instances for training.

Features:
- Whitelisted GPUs with known good performance/price
- Price tier limits (budget, standard, premium)
- Auto-selection of best available GPU
- Instance lifecycle management
"""

from __future__ import annotations

import os
import json
import re
import time
import subprocess
import urllib.request
import urllib.parse
from enum import Enum
from dataclasses import dataclass, field
from typing import Optional, Literal, Union, Dict, List, Tuple
from pathlib import Path


class GPUTier(Enum):
    """Price tiers for GPU selection."""
    BUDGET = "budget"      # < $0.10/hr - RTX 3060, 3070, 4060
    STANDARD = "standard"  # < $0.25/hr - RTX 3080, 3090, 4070, 4080
    PREMIUM = "premium"    # < $0.60/hr - RTX 4090, A10, A100


@dataclass
class GPUSpec:
    """GPU specification with performance characteristics."""
    name: str
    vram_gb: int
    tier: GPUTier
    max_price: float  # Maximum acceptable $/hr
    fp16_tflops: float  # Approximate FP16 performance
    priority: int  # Higher = preferred when price is similar


# Whitelisted GPUs with known good training performance
WHITELISTED_GPUS: Dict[str, GPUSpec] = {
    # Budget tier - Great for small models and testing
    "RTX_3060": GPUSpec("RTX 3060", 12, GPUTier.BUDGET, 0.08, 12.7, 1),
    "RTX_3070": GPUSpec("RTX 3070", 8, GPUTier.BUDGET, 0.10, 20.3, 2),
    "RTX_4060": GPUSpec("RTX 4060", 8, GPUTier.BUDGET, 0.10, 15.1, 3),
    "RTX_4060_Ti": GPUSpec("RTX 4060 Ti", 16, GPUTier.BUDGET, 0.12, 22.1, 4),

    # Standard tier - Good balance of price/performance
    "RTX_3080": GPUSpec("RTX 3080", 10, GPUTier.STANDARD, 0.15, 29.8, 5),
    "RTX_3080_Ti": GPUSpec("RTX 3080 Ti", 12, GPUTier.STANDARD, 0.18, 34.1, 6),
    "RTX_3090": GPUSpec("RTX 3090", 24, GPUTier.STANDARD, 0.20, 35.6, 7),
    "RTX_4070": GPUSpec("RTX 4070", 12, GPUTier.STANDARD, 0.15, 29.2, 8),
    "RTX_4070_Ti": GPUSpec("RTX 4070 Ti", 12, GPUTier.STANDARD, 0.18, 40.1, 9),
    "RTX_4080": GPUSpec("RTX 4080", 16, GPUTier.STANDARD, 0.25, 48.7, 10),

    # Premium tier - Maximum performance
    "RTX_4090": GPUSpec("RTX 4090", 24, GPUTier.PREMIUM, 0.45, 82.6, 11),
    "A10": GPUSpec("A10", 24, GPUTier.PREMIUM, 0.40, 31.2, 12),
    "A40": GPUSpec("A40", 48, GPUTier.PREMIUM, 0.55, 37.4, 13),
    "A100_40GB": GPUSpec("A100 40GB", 40, GPUTier.PREMIUM, 0.80, 77.9, 14),
    "A100_80GB": GPUSpec("A100 80GB", 80, GPUTier.PREMIUM, 1.20, 77.9, 15),
    "H100": GPUSpec("H100", 80, GPUTier.PREMIUM, 2.50, 267.0, 16),
}


@dataclass
class InstanceConfig:
    """Configuration for a Vast.ai instance."""
    image: str = "pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime"
    disk_gb: int = 50
    env_vars: dict = field(default_factory=dict)
    onstart_script: str = ""
    ssh_key: Optional[str] = None


@dataclass
class RunningInstance:
    """Information about a running instance."""
    instance_id: int
    gpu_name: str
    price_per_hour: float
    ssh_host: str
    ssh_port: int
    status: str
    gpu_util: float = 0.0
    disk_used_gb: float = 0.0


class VastAIProvider:
    """
    Vast.ai GPU provider with smart instance selection.

    Usage:
        provider = VastAIProvider(api_key="...", tier=GPUTier.STANDARD)

        # Find best available GPU
        offers = provider.find_offers(min_vram=12)

        # Create instance
        instance = provider.create_instance(
            offer_id=offers[0]["id"],
            config=InstanceConfig(image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime")
        )

        # Wait for ready
        provider.wait_for_ready(instance.instance_id)

        # Run training
        provider.run_command(instance.instance_id, "python train.py")

        # Cleanup
        provider.destroy_instance(instance.instance_id)
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        tier: GPUTier = GPUTier.STANDARD,
        max_price_override: Optional[float] = None,
    ):
        """
        Initialize Vast.ai provider.

        Args:
            api_key: Vast.ai API key (or set VAST_API_KEY env var)
            tier: Maximum GPU tier to consider
            max_price_override: Override max price regardless of tier
        """
        self.api_key = api_key or os.environ.get("VAST_API_KEY")
        if not self.api_key:
            raise ValueError("Vast.ai API key required (pass api_key or set VAST_API_KEY)")

        self.tier = tier
        self.max_price_override = max_price_override

        # Build allowed GPUs based on tier
        self._allowed_gpus = self._get_allowed_gpus()

    def _get_allowed_gpus(self) -> Dict[str, GPUSpec]:
        """Get GPUs allowed for current tier."""
        tier_order = [GPUTier.BUDGET, GPUTier.STANDARD, GPUTier.PREMIUM]
        max_tier_idx = tier_order.index(self.tier)

        return {
            name: spec
            for name, spec in WHITELISTED_GPUS.items()
            if tier_order.index(spec.tier) <= max_tier_idx
        }

    def _api_request(self, endpoint: str, method: str = "GET", data: Optional[dict] = None) -> Union[dict, list]:
        """Make a direct API request to Vast.ai."""
        base_url = "https://console.vast.ai/api/v0"
        url = f"{base_url}/{endpoint}"

        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        if data:
            headers["Content-Type"] = "application/json"
            req = urllib.request.Request(
                url,
                data=json.dumps(data).encode("utf-8"),
                headers=headers,
                method=method,
            )
        else:
            req = urllib.request.Request(url, headers=headers, method=method)

        try:
            with urllib.request.urlopen(req, timeout=30) as response:
                return json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8") if e.fp else ""
            raise RuntimeError(f"Vast.ai API error {e.code}: {error_body}")
        except urllib.error.URLError as e:
            raise RuntimeError(f"Vast.ai API connection error: {e.reason}")

    def _run_vastai(self, *args, parse_json: bool = True) -> Union[dict, str]:
        """Run vastai CLI command (legacy method, prefer _api_request)."""
        # Try common installation paths
        vastai_paths = [
            os.path.expanduser("~/Library/Python/3.9/bin/vastai"),
            os.path.expanduser("~/Library/Python/3.10/bin/vastai"),
            os.path.expanduser("~/Library/Python/3.11/bin/vastai"),
            os.path.expanduser("~/.local/bin/vastai"),
            "/usr/local/bin/vastai",
            "vastai",
        ]

        vastai_cmd = None
        for path in vastai_paths:
            if path == "vastai" or os.path.exists(path):
                vastai_cmd = path
                break

        if vastai_cmd is None:
            raise RuntimeError("vastai CLI not found. Install with: pip install vastai")

        # The key stays in the process environment. Putting it in argv
        # exposes it to every user who can read the process list.
        env = os.environ.copy()
        env["VAST_API_KEY"] = self.api_key
        cmd = [vastai_cmd] + list(args)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            env=env,
        )

        if result.returncode != 0:
            raise RuntimeError(f"vastai command failed: {result.stderr}")

        if parse_json:
            try:
                return json.loads(result.stdout)
            except json.JSONDecodeError:
                return result.stdout
        return result.stdout

    def _normalize_gpu_name(self, raw_name: str) -> Optional[str]:
        """Match a raw GPU name to our whitelist."""
        raw_lower = raw_name.lower().replace(" ", "").replace("-", "")

        mappings = {
            "rtx3060": "RTX_3060",
            "rtx3070": "RTX_3070",
            "rtx3080": "RTX_3080",
            "rtx3080ti": "RTX_3080_Ti",
            "rtx3090": "RTX_3090",
            "rtx4060": "RTX_4060",
            "rtx4060ti": "RTX_4060_Ti",
            "rtx4070": "RTX_4070",
            "rtx4070ti": "RTX_4070_Ti",
            "rtx4080": "RTX_4080",
            "rtx4090": "RTX_4090",
            "a10": "A10",
            "a40": "A40",
            "a10040gb": "A100_40GB",
            "a10080gb": "A100_80GB",
            "a100pcie40gb": "A100_40GB",
            "a100pcie80gb": "A100_80GB",
            "a100sxm440gb": "A100_40GB",
            "a100sxm480gb": "A100_80GB",
            "h100": "H100",
        }

        for pattern, gpu_key in mappings.items():
            if pattern in raw_lower:
                return gpu_key

        return None

    def get_max_price(self) -> float:
        """Get maximum price for current configuration."""
        if self.max_price_override:
            return self.max_price_override

        # Use highest price in allowed tier
        return max(spec.max_price for spec in self._allowed_gpus.values())

    def find_offers(
        self,
        min_vram: int = 8,
        min_reliability: float = 0.95,
        min_bandwidth: int = 100,
        sort_by: Literal["price", "performance", "value"] = "value",
        limit: int = 10,
    ) -> List[dict]:
        """
        Find available GPU offers matching criteria.

        Args:
            min_vram: Minimum VRAM in GB
            min_reliability: Minimum reliability score (0-1)
            min_bandwidth: Minimum internet bandwidth in Mbps
            sort_by: How to rank results
            limit: Maximum results to return

        Returns:
            List of offers sorted by preference
        """
        max_price = self.get_max_price()

        # Build query for API - Vast.ai uses JSON query format
        query = {
            "verified": {"eq": True},
            "external": {"eq": False},
            "rentable": {"eq": True},
            "num_gpus": {"eq": 1},
            "gpu_ram": {"gte": min_vram * 1024},  # API uses MB
            "reliability2": {"gte": min_reliability},
            "inet_down": {"gte": min_bandwidth},
            "dph_total": {"lte": max_price},
            "order": [["dph_total", "asc"]],
            "type": "on-demand",
        }

        try:
            # URL encode the query
            query_str = urllib.parse.quote(json.dumps(query))
            result = self._api_request(f"bundles?q={query_str}")
            offers = result.get("offers", []) if isinstance(result, dict) else result
        except Exception as e:
            print(f"Warning: Vast.ai API search failed: {e}")
            return []

        if not offers:
            return []

        # Filter to whitelisted GPUs
        filtered = []
        for offer in offers:
            gpu_name = offer.get("gpu_name", "")
            normalized = self._normalize_gpu_name(gpu_name)

            if normalized and normalized in self._allowed_gpus:
                spec = self._allowed_gpus[normalized]
                offer["_gpu_spec"] = spec
                offer["_normalized_name"] = normalized

                # Calculate value score (performance per dollar)
                price = offer.get("dph_total", 1.0)
                offer["_value_score"] = spec.fp16_tflops / price if price > 0 else 0

                filtered.append(offer)

        # Sort by preference
        if sort_by == "price":
            filtered.sort(key=lambda x: x.get("dph_total", 999))
        elif sort_by == "performance":
            filtered.sort(key=lambda x: -x["_gpu_spec"].fp16_tflops)
        else:  # value
            filtered.sort(key=lambda x: -x["_value_score"])

        return filtered[:limit]

    def find_best_offer(
        self,
        min_vram: int = 8,
        prefer_value: bool = True,
    ) -> Optional[dict]:
        """Find the single best offer for current requirements."""
        offers = self.find_offers(
            min_vram=min_vram,
            sort_by="value" if prefer_value else "price",
            limit=1,
        )
        return offers[0] if offers else None

    def create_instance(
        self,
        offer_id: int,
        config: Optional[InstanceConfig] = None,
    ) -> RunningInstance:
        """
        Create a new instance from an offer.

        Args:
            offer_id: The offer ID to create instance from
            config: Instance configuration

        Returns:
            RunningInstance with connection details
        """
        config = config or InstanceConfig()

        # Build API request payload
        payload = {
            "client_id": "me",
            "image": config.image,
            "disk": config.disk_gb,
            "runtype": "ssh",  # SSH access
        }

        if config.onstart_script:
            payload["onstart"] = config.onstart_script

        if config.env_vars:
            payload["env"] = config.env_vars

        try:
            result = self._api_request(f"asks/{offer_id}/", method="PUT", data=payload)
        except Exception as e:
            raise RuntimeError(f"Failed to create instance: {e}")

        # API returns {"success": true, "new_contract": 12345}
        if isinstance(result, dict):
            if result.get("success"):
                instance_id = result.get("new_contract", 0)
            else:
                raise RuntimeError(f"Instance creation failed: {result}")
        else:
            raise RuntimeError(f"Unexpected response: {result}")

        return RunningInstance(
            instance_id=instance_id,
            gpu_name="",
            price_per_hour=0.0,
            ssh_host="",
            ssh_port=0,
            status="starting",
        )

    def get_instance(self, instance_id: int) -> Optional[RunningInstance]:
        """Get details of a running instance."""
        try:
            instances = self._run_vastai("show", "instances", "--raw")
        except Exception:
            return None

        if isinstance(instances, str):
            return None

        for inst in instances:
            if inst.get("id") == instance_id:
                return RunningInstance(
                    instance_id=instance_id,
                    gpu_name=inst.get("gpu_name", ""),
                    price_per_hour=inst.get("dph_total", 0.0),
                    ssh_host=inst.get("ssh_host", ""),
                    ssh_port=inst.get("ssh_port", 0),
                    status=inst.get("actual_status", "unknown"),
                    gpu_util=inst.get("gpu_util", 0.0),
                    disk_used_gb=inst.get("disk_usage", 0.0),
                )

        return None

    def list_instances(self) -> List[RunningInstance]:
        """List all running instances."""
        try:
            result = self._api_request("instances?owner=me")
            instances = result.get("instances", []) if isinstance(result, dict) else result
        except Exception:
            return []

        if not instances:
            return []

        return [
            RunningInstance(
                instance_id=inst.get("id", 0),
                gpu_name=inst.get("gpu_name", ""),
                price_per_hour=inst.get("dph_total", 0.0),
                ssh_host=inst.get("public_ipaddr", ""),
                ssh_port=inst.get("ssh_port", 0),
                status=inst.get("actual_status", "unknown"),
                gpu_util=inst.get("gpu_util", 0.0),
                disk_used_gb=inst.get("disk_usage", 0.0),
            )
            for inst in instances
        ]

    def wait_for_ready(
        self,
        instance_id: int,
        timeout: int = 300,
        poll_interval: int = 10,
    ) -> RunningInstance:
        """
        Wait for instance to be ready for SSH.

        Args:
            instance_id: Instance to wait for
            timeout: Maximum seconds to wait
            poll_interval: Seconds between status checks

        Returns:
            Running instance with connection details
        """
        start = time.time()

        while time.time() - start < timeout:
            instance = self.get_instance(instance_id)

            if instance and instance.status == "running" and instance.ssh_port > 0:
                return instance

            status = instance.status if instance else "not found"
            print(f"Instance {instance_id}: {status}, waiting...")
            time.sleep(poll_interval)

        raise TimeoutError(f"Instance {instance_id} not ready after {timeout}s")

    def run_command(
        self,
        instance_id: int,
        command: str,
        timeout: int = 3600,
    ) -> Tuple[int, str, str]:
        """
        Run a command on an instance via SSH.

        Returns:
            (return_code, stdout, stderr)
        """
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError(f"Instance {instance_id} not ready for SSH")

        ssh_cmd = [
            "ssh",
            "-o", "StrictHostKeyChecking=yes",
            "-p", str(instance.ssh_port),
            f"root@{instance.ssh_host}",
            command,
        ]

        result = subprocess.run(
            ssh_cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

        return result.returncode, result.stdout, result.stderr

    def copy_to_instance(
        self,
        instance_id: int,
        local_path: str,
        remote_path: str,
    ):
        """Copy files to instance via SCP."""
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError(f"Instance {instance_id} not ready for SCP")

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=yes",
            "-P", str(instance.ssh_port),
            "-r",
            local_path,
            f"root@{instance.ssh_host}:{remote_path}",
        ]

        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

    def copy_from_instance(
        self,
        instance_id: int,
        remote_path: str,
        local_path: str,
    ):
        """Copy files from instance via SCP."""
        instance = self.get_instance(instance_id)
        if not instance or not instance.ssh_host:
            raise RuntimeError(f"Instance {instance_id} not ready for SCP")

        scp_cmd = [
            "scp",
            "-o", "StrictHostKeyChecking=yes",
            "-P", str(instance.ssh_port),
            "-r",
            f"root@{instance.ssh_host}:{remote_path}",
            local_path,
        ]

        result = subprocess.run(scp_cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"SCP failed: {result.stderr}")

    def destroy_instance(self, instance_id: int):
        """Terminate and destroy an instance."""
        if not str(instance_id).isdigit():
            raise ValueError("instance id must be numeric")
        self._run_vastai("destroy", "instance", str(instance_id), parse_json=False)

    def get_account_balance(self) -> float:
        """Get current account balance (credit + balance)."""
        try:
            result = self._api_request("users/current")
            if isinstance(result, dict):
                # Vast.ai uses 'credit' for prepaid credit and 'balance' for account balance
                credit = float(result.get("credit", 0.0))
                balance = float(result.get("balance", 0.0))
                return credit + balance
        except Exception:
            pass
        return 0.0

    def estimate_cost(self, price_per_hour: float, hours: float) -> dict:
        """Estimate cost for a training run."""
        cost = price_per_hour * hours
        balance = self.get_account_balance()

        return {
            "price_per_hour": price_per_hour,
            "estimated_hours": hours,
            "total_cost": cost,
            "current_balance": balance,
            "remaining_after": balance - cost,
            "can_afford": balance >= cost,
        }


def quick_train(
    training_script: str,
    min_vram: int = 12,
    tier: GPUTier = GPUTier.STANDARD,
    api_key: Optional[str] = None,
    output_dir: str = "/workspace/output",
) -> dict:
    """
    Quick helper to run training on Vast.ai.

    Args:
        training_script: Path to training script (will be copied)
        min_vram: Minimum GPU VRAM required
        tier: GPU price tier
        api_key: Vast.ai API key
        output_dir: Remote directory for outputs

    Returns:
        Dict with instance info and output paths
    """
    provider = VastAIProvider(api_key=api_key, tier=tier)

    # Find best offer
    offer = provider.find_best_offer(min_vram=min_vram)
    if not offer:
        raise RuntimeError(f"No GPUs available meeting requirements (vram>={min_vram}GB, tier={tier})")

    print(f"Found: {offer['gpu_name']} @ ${offer['dph_total']:.3f}/hr")

    # Create instance
    config = InstanceConfig(
        image="pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime",
        disk_gb=50,
        onstart_script=f"pip install einops && mkdir -p {output_dir}",
    )

    instance = provider.create_instance(offer["id"], config)
    print(f"Created instance {instance.instance_id}")

    # Wait for ready
    instance = provider.wait_for_ready(instance.instance_id)
    print(f"Instance ready: {instance.ssh_host}:{instance.ssh_port}")

    return {
        "provider": provider,
        "instance": instance,
        "offer": offer,
        "output_dir": output_dir,
    }
