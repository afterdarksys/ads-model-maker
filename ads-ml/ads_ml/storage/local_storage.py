"""
Local filesystem storage backend.

Used for development and as a fallback when cloud storage is not configured.
"""

import os
import shutil
from pathlib import Path
from typing import Iterator

from ads_ml.storage.base import StorageBackend


class LocalStorage(StorageBackend):
    """
    Local filesystem storage backend.

    Environment variables:
        LOCAL_STORAGE_PATH: Base directory for storage (default: ./data)
    """

    def __init__(self, base_path: str | Path | None = None):
        base_path = base_path or os.environ.get("LOCAL_STORAGE_PATH", "./data")
        self.base_path = Path(base_path).resolve()
        self.base_path.mkdir(parents=True, exist_ok=True)

    def _resolve_path(self, remote_path: str) -> Path:
        """Resolve remote path to local path safely.

        A string prefix check treats ``/data`` as a parent of ``/data-evil``.
        Compare path parents instead.
        """
        if remote_path is None:
            raise ValueError("path is required")
        base = self.base_path.resolve()
        candidate = (base / remote_path).resolve()
        if candidate != base and base not in candidate.parents:
            raise ValueError(f"Invalid path: {remote_path}")
        return candidate

    def upload_file(self, local_path: Path, remote_path: str) -> str:
        """Copy file to local storage."""
        dest = self._resolve_path(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return f"file://{dest}"

    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Copy file from local storage."""
        src = self._resolve_path(remote_path)
        local_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)

    def upload_bytes(self, data: bytes, remote_path: str) -> str:
        """Write bytes to local storage."""
        dest = self._resolve_path(remote_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        return f"file://{dest}"

    def download_bytes(self, remote_path: str) -> bytes:
        """Read bytes from local storage."""
        src = self._resolve_path(remote_path)
        return src.read_bytes()

    def list_objects(self, prefix: str = "") -> list[str]:
        """List files in local storage."""
        search_path = self._resolve_path(prefix) if prefix else self.base_path
        if search_path.is_file():
            return [prefix]

        results = []
        if search_path.exists():
            for path in search_path.rglob("*"):
                if path.is_file():
                    rel_path = path.relative_to(self.base_path)
                    results.append(str(rel_path))
        return results

    def delete_object(self, remote_path: str) -> bool:
        """Delete file from local storage."""
        try:
            path = self._resolve_path(remote_path)
            path.unlink()
            return True
        except (FileNotFoundError, PermissionError):
            return False

    def exists(self, remote_path: str) -> bool:
        """Check if file exists in local storage."""
        return self._resolve_path(remote_path).exists()

    def get_url(self, remote_path: str, expiry_seconds: int = 3600) -> str:
        """Get file:// URL for local file."""
        return f"file://{self._resolve_path(remote_path)}"

    def stream_lines(self, remote_path: str, encoding: str = "utf-8") -> Iterator[str]:
        """Stream lines from a text file."""
        path = self._resolve_path(remote_path)
        with open(path, "r", encoding=encoding, errors="replace") as f:
            for line in f:
                yield line.rstrip("\n\r")
