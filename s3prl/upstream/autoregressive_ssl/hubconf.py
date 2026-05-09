"""
Hub configuration for autoregressive SSL upstream models
"""

import os
from .expert import UpstreamExpert


# pylint: disable=W1113
def autoregressive_wav2vec2_local(ckpt, *args, **kwargs):
    """
    Load local Wav2Vec2 model with autoregressive masking

    Args:
        ckpt: Local path to the model checkpoint
    """
    assert os.path.isdir(ckpt), f"Checkpoint path {ckpt} does not exist"
    return UpstreamExpert(ckpt, model_type="wav2vec2", *args, **kwargs)


def autoregressive_wav2vec2_hf(ckpt="yky-h/japanese-wav2vec2-base", *args, **kwargs):
    """
    Load HuggingFace Wav2Vec2 model with autoregressive masking

    Args:
        ckpt: HuggingFace model ID (default: yky-h/japanese-wav2vec2-base)
    """
    return UpstreamExpert(ckpt, model_type="wav2vec2", *args, **kwargs)


def autoregressive_hubert_local(ckpt, *args, **kwargs):
    """
    Load local HuBERT model with autoregressive masking

    Args:
        ckpt: Local path to the model checkpoint
    """
    assert os.path.isdir(ckpt), f"Checkpoint path {ckpt} does not exist"
    return UpstreamExpert(ckpt, model_type="hubert", *args, **kwargs)


def autoregressive_hubert_hf(ckpt="yky-h/japanese-hubert-base", *args, **kwargs):
    """
    Load HuggingFace HuBERT model with autoregressive masking

    Args:
        ckpt: HuggingFace model ID (default: yky-h/japanese-hubert-base)
    """
    return UpstreamExpert(ckpt, model_type="hubert", *args, **kwargs)


# Default entry points
def autoregressive_wav2vec2(*args, **kwargs):
    """Default Wav2Vec2 with autoregressive masking"""
    return autoregressive_wav2vec2_hf(*args, **kwargs)


def autoregressive_hubert(*args, **kwargs):
    """Default HuBERT with autoregressive masking"""
    return autoregressive_hubert_hf(*args, **kwargs)
