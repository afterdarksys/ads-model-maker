"""
Oracle Cloud Infrastructure Object Storage backend.

Supports both instance principal auth (for OCI compute) and config file auth.
"""

import os
from pathlib import Path
from typing import Iterator

import oci
from oci.object_storage import ObjectStorageClient
from oci.object_storage.models import CreatePreauthenticatedRequestDetails

from ads_ml.storage.base import StorageBackend


class OCIStorage(StorageBackend):
    """
    OCI Object Storage backend.

    Environment variables:
        OCI_NAMESPACE: Object storage namespace (auto-detected if not set)
        OCI_BUCKET: Bucket name (required)
        OCI_REGION: Region (e.g., us-ashburn-1)
        OCI_CONFIG_PATH: Path to OCI config file (optional)
        OCI_PROFILE: OCI config profile name (default: DEFAULT)
        OCI_AUTH: Auth method - 'instance_principal' or 'config' (default: auto)
    """

    def __init__(
        self,
        bucket: str | None = None,
        namespace: str | None = None,
        region: str | None = None,
        auth_type: str | None = None,
    ):
        self.bucket = bucket or os.environ.get("OCI_BUCKET")
        if not self.bucket:
            raise ValueError("Bucket name required via OCI_BUCKET env var or constructor")

        self.region = region or os.environ.get("OCI_REGION", "us-ashburn-1")
        auth_type = auth_type or os.environ.get("OCI_AUTH", "auto")

        # Initialize client based on auth type
        if auth_type == "instance_principal" or (
            auth_type == "auto" and self._is_oci_instance()
        ):
            self._init_instance_principal()
        else:
            self._init_config_auth()

        # Get or set namespace
        self.namespace = namespace or os.environ.get("OCI_NAMESPACE")
        if not self.namespace:
            self.namespace = self.client.get_namespace().data

    def _is_oci_instance(self) -> bool:
        """Check if running on OCI instance (metadata service available)."""
        try:
            import urllib.request

            req = urllib.request.Request(
                "http://169.254.169.254/opc/v2/instance/",
                headers={"Authorization": "Bearer Oracle"},
            )
            urllib.request.urlopen(req, timeout=1)
            return True
        except Exception:
            return False

    def _init_instance_principal(self) -> None:
        """Initialize client with instance principal authentication."""
        signer = oci.auth.signers.InstancePrincipalsSecurityTokenSigner()
        self.client = ObjectStorageClient(config={}, signer=signer)

    def _init_config_auth(self) -> None:
        """Initialize client with config file authentication."""
        config_path = os.environ.get("OCI_CONFIG_PATH", "~/.oci/config")
        profile = os.environ.get("OCI_PROFILE", "DEFAULT")
        config = oci.config.from_file(config_path, profile)
        self.client = ObjectStorageClient(config)

    def upload_file(self, local_path: Path, remote_path: str) -> str:
        """Upload file to OCI Object Storage."""
        with open(local_path, "rb") as f:
            self.client.put_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=remote_path,
                put_object_body=f,
            )
        return f"oci://{self.bucket}/{remote_path}"

    def download_file(self, remote_path: str, local_path: Path) -> None:
        """Download file from OCI Object Storage."""
        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=remote_path,
        )
        local_path.parent.mkdir(parents=True, exist_ok=True)
        with open(local_path, "wb") as f:
            for chunk in response.data.raw.stream(1024 * 1024, decode_content=False):
                f.write(chunk)

    def upload_bytes(self, data: bytes, remote_path: str) -> str:
        """Upload bytes to OCI Object Storage."""
        self.client.put_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=remote_path,
            put_object_body=data,
        )
        return f"oci://{self.bucket}/{remote_path}"

    def download_bytes(self, remote_path: str) -> bytes:
        """Download bytes from OCI Object Storage."""
        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=remote_path,
        )
        return response.data.content

    def list_objects(self, prefix: str = "") -> list[str]:
        """List objects with optional prefix filter."""
        objects = []
        next_start = None

        while True:
            response = self.client.list_objects(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                prefix=prefix,
                start=next_start,
                limit=1000,
            )
            objects.extend([obj.name for obj in response.data.objects])

            if response.data.next_start_with:
                next_start = response.data.next_start_with
            else:
                break

        return objects

    def delete_object(self, remote_path: str) -> bool:
        """Delete an object from OCI Object Storage."""
        try:
            self.client.delete_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=remote_path,
            )
            return True
        except oci.exceptions.ServiceError:
            return False

    def exists(self, remote_path: str) -> bool:
        """Check if object exists in OCI Object Storage."""
        try:
            self.client.head_object(
                namespace_name=self.namespace,
                bucket_name=self.bucket,
                object_name=remote_path,
            )
            return True
        except oci.exceptions.ServiceError:
            return False

    def get_url(self, remote_path: str, expiry_seconds: int = 3600) -> str:
        """Get pre-authenticated request URL for object."""
        import datetime

        details = CreatePreauthenticatedRequestDetails(
            name=f"par-{remote_path.replace('/', '-')}",
            access_type="ObjectRead",
            time_expires=datetime.datetime.utcnow()
            + datetime.timedelta(seconds=expiry_seconds),
            object_name=remote_path,
        )

        response = self.client.create_preauthenticated_request(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            create_preauthenticated_request_details=details,
        )

        # Construct full URL
        base_url = f"https://objectstorage.{self.region}.oraclecloud.com"
        return f"{base_url}{response.data.access_uri}"

    def stream_lines(
        self, remote_path: str, encoding: str = "utf-8", chunk_size: int = 10 * 1024 * 1024
    ) -> Iterator[str]:
        """
        Stream lines from a text file in OCI Object Storage.

        Memory-efficient for large files - processes in chunks.
        """
        response = self.client.get_object(
            namespace_name=self.namespace,
            bucket_name=self.bucket,
            object_name=remote_path,
        )

        buffer = ""
        for chunk in response.data.raw.stream(chunk_size, decode_content=False):
            buffer += chunk.decode(encoding, errors="replace")
            lines = buffer.split("\n")
            # Yield all complete lines
            for line in lines[:-1]:
                yield line
            # Keep incomplete line in buffer
            buffer = lines[-1]

        # Yield final line if exists
        if buffer:
            yield buffer

    def stream_all_files(
        self, prefix: str = "", encoding: str = "utf-8"
    ) -> Iterator[tuple[str, Iterator[str]]]:
        """
        Stream lines from all files matching prefix.

        Yields: (filename, line_iterator) tuples
        """
        for obj_name in self.list_objects(prefix):
            yield obj_name, self.stream_lines(obj_name, encoding)
