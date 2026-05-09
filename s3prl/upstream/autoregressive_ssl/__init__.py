"""
Autoregressive SSL Upstream Models with Chunk-wise Causal Mask
"""

from .hubconf import (
    autoregressive_wav2vec2,
    autoregressive_wav2vec2_local,
    autoregressive_wav2vec2_hf,
    autoregressive_hubert,
    autoregressive_hubert_local,
    autoregressive_hubert_hf,
)

__all__ = [
    "autoregressive_wav2vec2",
    "autoregressive_wav2vec2_local",
    "autoregressive_wav2vec2_hf",
    "autoregressive_hubert",
    "autoregressive_hubert_local",
    "autoregressive_hubert_hf",
]
