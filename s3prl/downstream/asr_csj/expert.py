import logging
from typing import Dict, List
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .dataset import CSJDataset

# from torch.nn.utils.rnn import pad_sequence
# from ..model import *

# from ..runner import Runner
# pylint: disable=W0613, W1203

LOGGER = logging.getLogger(__name__)


class DownstreamExpert(nn.Module):
    """
    Downstream expert for CSJ ASR using CTC loss
    Enhanced with configurable hyperparameters
    """

    def __init__(self, upstream_dim: int, downstream_expert: Dict, expdir: str, **kwargs):
        """
        Args:
            upstream_dim: Dimension of upstream model output
            downstream_expert: Config dictionary for downstream model
            expdir: Experiment directory
        """
        super().__init__()

        self.upstream_dim = upstream_dim
        self.datarc = downstream_expert["datarc"]
        self.modelrc = downstream_expert["modelrc"]

        # Get vocab size from config
        self.vocab_size = self.modelrc.get("vocab_size", 3000)

        # CTC ASR head: linear projection from upstream_dim to vocab_size
        self.projector = nn.Linear(upstream_dim, self.vocab_size)

        # CTC loss configuration
        ctc_config = self.modelrc.get("ctc_loss", {})
        self.ctc_loss = nn.CTCLoss(
            blank=ctc_config.get("blank", 0),
            reduction=ctc_config.get("reduction", "mean"),
            zero_infinity=ctc_config.get("zero_infinity", True),
        )

        self.register_buffer("best_score", torch.zeros(1))

    def get_dataloader(self, mode: str, epoch: int = 0) -> DataLoader:
        """
        Get dataloader for train/dev/test

        Args:
            mode: 'train', 'dev', or 'test'
            epoch: Current epoch number

        Returns:
            DataLoader object
        """
        if mode == "train":
            manifest_path = self.datarc["train_manifest"]
            batch_size = self.datarc.get("train_batch_size", 16)
            shuffle = True
        elif mode == "dev":
            manifest_path = self.datarc["valid_manifest"]
            batch_size = self.datarc.get("eval_batch_size", 8)
            shuffle = False
        elif mode == "test":
            manifest_path = self.datarc["test_manifest"]
            batch_size = self.datarc.get("eval_batch_size", 8)
            shuffle = False
        else:
            raise ValueError(f"Unknown mode: {mode}")

        # SpecAugment configuration
        spec_augment_config = self.datarc.get("spec_augment", {})
        use_spec_augment = spec_augment_config.get("enabled", False) and mode == "train"

        dataset = CSJDataset(
            manifest_path=manifest_path,
            vocab_size=self.vocab_size,
            use_spec_augment=use_spec_augment,
            spec_augment_config=spec_augment_config if use_spec_augment else None,
            **self.datarc,
        )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=dataset.collate_fn,
            num_workers=self.datarc.get("num_workers", 4),
        )

    def forward(self, mode: str, features: List[torch.Tensor], labels: torch.Tensor, records: Dict, **kwargs):
        """
        Forward pass for training/validation

        Args:
            mode: 'train' or 'dev'
            features: List of feature tensors from upstream [B, T, D]
            labels: Label tensor [B, L]
            records: Dictionary to store metrics

        Returns:
            Loss value
        """
        # features is a list of tensors from different layers
        # Use the last layer by default
        features = features[-1] if isinstance(features, list) else features

        # Project to vocabulary space
        logits = self.projector(features)  # [B, T, V]

        # Prepare for CTC loss
        log_probs = nn.functional.log_softmax(logits, dim=-1)  # [B, T, V]
        log_probs = log_probs.transpose(0, 1)  # [T, B, V] for CTC

        # Get lengths
        feature_lengths = torch.sum(
            kwargs.get("feature_len", torch.ones(features.size(0), features.size(1))), dim=1
        ).long()
        label_lengths = torch.sum(labels != -100, dim=1).long()

        # Filter out -100 padding in labels
        labels_masked = labels.clone()
        labels_masked[labels_masked == -100] = 0

        # Compute CTC loss
        loss = self.ctc_loss(log_probs, labels_masked, feature_lengths, label_lengths)

        records["loss"] = loss.item()
        records["ctc_loss"] = loss.item()

        return loss

    def log_records(
        self,
        mode: str,
        records: Dict,
        logger: logging.Logger = None,
        global_step: int = 0,
        batch_ids: List = None,
        total_batch_num: int = 0,
        **kwargs,
    ):
        """Log training/validation metrics"""
        logger = logger if logger is not None else LOGGER
        if mode == "train":
            logger.info(f"[Train] Step {global_step}/{total_batch_num}: " f"Loss = {records['loss']:.4f}")
        else:
            avg_loss = (
                sum(records["loss"]) / len(records["loss"]) if isinstance(records["loss"], list) else records["loss"]
            )
            logger.info(f"[{mode.capitalize()}] Loss = {avg_loss:.4f}")
