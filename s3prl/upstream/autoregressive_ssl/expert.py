import logging
from typing import List, Union
import torch
import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel, Wav2Vec2Model, HubertModel

# Import from the provided autoregressive_ssl.py
from .tools.autoregressive_ssl import overwrite_model_for_autoregressive, Wav2Vec2ForCTCSRM, HubertForCTCSRM
from .tools.cnn_tools import calc_stride_of_receptive_field

# from s3prl.s3prl.upstream.autoregressive_ssl.autoregressive_ssl import overwrite_model_for_autoregressive
# from s3prl.s3prl.upstream.autoregressive_ssl.cnn_tools import calc_stride_of_receptive_field

logger = logging.getLogger(__name__)

# pylint: disable=W1203


class UpstreamExpert(nn.Module):
    """
    Autoregressive SSL Upstream Expert for S3PRL
    Supports Wav2Vec2 and HuBERT with chunk-wise causal masking
    """

    def __init__(
        self,
        ckpt: str,
        model_type: str = "wav2vec2",
        chunk_size: int = None,
        sample_rate: int = 16000,
        down_sample_target_fps: float = 12.5,
        freeze_cnn: bool = True,
        **kwargs,
    ):
        """
        Args:
            ckpt: Path or HuggingFace model ID for the pretrained model
            model_type: 'wav2vec2' or 'hubert'
            chunk_size: Optional manual chunk size (if None, auto-calculated)
            sample_rate: Audio sampling rate (default: 16000)
            down_sample_target_fps: Target frame rate for downsampling (default: 12.5)
            freeze_cnn: Whether to freeze CNN feature extractor (default: True)
        """
        super().__init__()

        self.ckpt = ckpt
        self.model_type = model_type.lower()
        self.sample_rate = sample_rate
        self.down_sample_target_fps = down_sample_target_fps
        self.freeze_cnn = freeze_cnn

        # Load feature extractor
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(ckpt)

        # Load base model
        logger.info(f"Loading {model_type} model from {ckpt}")
        self.model: Union[Wav2Vec2Model, HubertModel] = AutoModel.from_pretrained(ckpt)

        # Calculate chunk_size if not provided
        if chunk_size is None:
            chunk_size = self._calculate_chunk_size()

        self.chunk_size = chunk_size
        logger.info(f"Chunk size: {chunk_size}")

        # Apply monkey patch for autoregressive behavior
        self.model: Union[Wav2Vec2ForCTCSRM, HubertForCTCSRM] = overwrite_model_for_autoregressive(
            self.model, chunk_size=self.chunk_size, freeze_cnn=self.freeze_cnn, wo_lm_head=True
        )

    def _calculate_chunk_size(self) -> int:
        """
        Calculate chunk_size based on model config and target FPS
        Following the formula from foundation_extractor3.py:
        chunk_size = (sample_rate / stride) / down_sample_target_fps
        """
        config = self.model.config

        # Get conv_stride from model config
        conv_stride = config.conv_stride

        # Calculate total stride
        stride = calc_stride_of_receptive_field(conv_stride, conv_type="conv1d")

        # Calculate audio feature FPS
        audio_feature_fps = self.sample_rate / stride

        # Calculate chunk_size
        chunk_size = audio_feature_fps / self.down_sample_target_fps

        # Validate that chunk_size is an integer
        if not chunk_size.is_integer():
            raise ValueError(
                f"chunk_size must be integer, but got "
                f"({self.sample_rate} / {stride}) / {self.down_sample_target_fps} = {chunk_size}"
            )

        return int(chunk_size)

    def get_downsample_rates(self, key: str = None) -> int:
        """Return the downsample rate of the model"""
        config = self.model.config
        conv_stride = config.conv_stride
        stride = calc_stride_of_receptive_field(conv_stride, conv_type="conv1d")
        return stride

    def forward(
        self,
        wavs: List[torch.Tensor],
        wav_lens: torch.Tensor = None,
        past_key_values: List = None,
        is_autoregressive: bool = True,
        **kwargs,
    ):
        """
        Forward pass with autoregressive masking support

        Args:
            wavs: List of waveform tensors or batched tensor [B, T]
            wav_lens: Optional length tensor [B]
            past_key_values: Optional cached key-value pairs for autoregressive inference
            is_autoregressive: Whether to apply autoregressive masking

        Returns:
            Dictionary containing:
                - hidden_states: List of hidden states from all layers
                - last_hidden_state: Final hidden state [B, T, D]
                - key_value_cache: Cached key-value pairs
        """
        # Prepare input
        if isinstance(wavs, list):
            wavs = [w.cpu().numpy() for w in wavs]
        elif isinstance(wavs, torch.Tensor):
            if wavs.dim() == 1:
                wavs = [wavs.cpu().numpy()]
            else:
                wavs = [w.cpu().numpy() for w in wavs]

        # Process with feature extractor
        inputs = self.feature_extractor(
            wavs,
            return_tensors="pt",
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
        )

        # Move to same device as model
        device = next(self.model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Forward through model
        outputs = self.model(
            **inputs,
            past_key_values=past_key_values,
            is_autoregressive=is_autoregressive,
            output_hidden_states=True,
            return_dict=True,
        )

        # Prepare S3PRL-compatible output
        # S3PRL expects a dictionary with "hidden_states" key containing all layer outputs
        hidden_states = outputs.hidden_states if hasattr(outputs, "hidden_states") else None
        last_hidden_state = outputs.last_hidden_state
        key_value_cache = outputs.key_value_cache if hasattr(outputs, "key_value_cache") else None

        return {
            "hidden_states": hidden_states,  # Tuple of all layer outputs
            "last_hidden_state": last_hidden_state,  # [B, T, D]
            "key_value_cache": key_value_cache,  # For autoregressive inference
        }
