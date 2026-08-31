from .base import BaseAttnBackend, BaseAttnMetadata, HybridBackend
from .fi import FICaptureData, FIMetadata, FlashInferBackend

# Compatibility aliases for course code written before the mini-sglang naming
# alignment. New code should use the short mini-sglang names above.
BaseAttentionBackend = BaseAttnBackend
BaseAttentionMetadata = BaseAttnMetadata
HybridAttentionBackend = HybridBackend
FlashInferAttentionBackend = FlashInferBackend
FlashInferAttentionMetadata = FIMetadata

__all__ = [
    "BaseAttnBackend",
    "BaseAttnMetadata",
    "HybridBackend",
    "FICaptureData",
    "FIMetadata",
    "FlashInferBackend",
]
