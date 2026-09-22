"""
ADS Storage Backends

Support for OCI Object Storage, S3-compatible storage, and local filesystem.
"""

from ads_ml.storage.base import StorageBackend
from ads_ml.storage.oci_storage import OCIStorage
from ads_ml.storage.local_storage import LocalStorage

__all__ = [
    "StorageBackend",
    "OCIStorage",
    "LocalStorage",
]
