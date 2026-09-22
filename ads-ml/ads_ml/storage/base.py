"""
Abstract base class for storage backends.
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import BinaryIO, Iterator


class StorageBackend(ABC):
    """Abstract storage backend interface for model artifacts and datasets."""

    @abstractmethod
    def upload_file(self, local_path: Path, remote_path: str) -> str:
        """
        Upload a file to remote storage.

        Args:
            local_path: Local file path
            remote_path: Remote object key/path

        Returns:
            Remote URL or URI of uploaded file
        """
        pass

    @abstractmethod
    def download_file(self, remote_path: str, local_path: Path) -> None:
        """
        Download a file from remote storage.

        Args:
            remote_path: Remote object key/path
            local_path: Local destination path
        """
        pass

    @abstractmethod
    def upload_bytes(self, data: bytes, remote_path: str) -> str:
        """
        Upload bytes directly to remote storage.

        Args:
            data: Bytes to upload
            remote_path: Remote object key/path

        Returns:
            Remote URL or URI
        """
        pass

    @abstractmethod
    def download_bytes(self, remote_path: str) -> bytes:
        """
        Download bytes from remote storage.

        Args:
            remote_path: Remote object key/path

        Returns:
            Downloaded bytes
        """
        pass

    @abstractmethod
    def list_objects(self, prefix: str = "") -> list[str]:
        """
        List objects with optional prefix filter.

        Args:
            prefix: Filter prefix

        Returns:
            List of object keys
        """
        pass

    @abstractmethod
    def delete_object(self, remote_path: str) -> bool:
        """
        Delete an object from storage.

        Args:
            remote_path: Remote object key/path

        Returns:
            True if deleted successfully
        """
        pass

    @abstractmethod
    def exists(self, remote_path: str) -> bool:
        """
        Check if an object exists.

        Args:
            remote_path: Remote object key/path

        Returns:
            True if object exists
        """
        pass

    @abstractmethod
    def get_url(self, remote_path: str, expiry_seconds: int = 3600) -> str:
        """
        Get a pre-signed URL for an object.

        Args:
            remote_path: Remote object key/path
            expiry_seconds: URL expiry time in seconds

        Returns:
            Pre-signed URL
        """
        pass

    def stream_lines(self, remote_path: str, encoding: str = "utf-8") -> Iterator[str]:
        """
        Stream lines from a text file in storage.

        Args:
            remote_path: Remote object key/path
            encoding: Text encoding

        Yields:
            Lines from the file
        """
        data = self.download_bytes(remote_path)
        for line in data.decode(encoding).splitlines():
            yield line
