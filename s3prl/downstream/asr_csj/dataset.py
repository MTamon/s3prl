import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import torch
import torchaudio
from torch.nn.utils.rnn import pad_sequence
import pickle

logger = logging.getLogger(__name__)


class CSJDataset(torch.utils.data.Dataset):
    """
    CSJ (Corpus of Spontaneous Japanese) Dataset
    Reads NeMo manifest format (JSON Lines)
    Automatically builds vocabulary from training data
    """

    def __init__(
        self,
        manifest_path: str,
        vocab_path: Optional[str] = None,  # 新規: vocabファイルのパス
        vocab_size: Optional[int] = None,  # Noneの場合は自動計算
        max_audio_len: float = 16.0,
        use_spec_augment: bool = False,
        spec_augment_config: Optional[Dict] = None,
        **kwargs,
    ):
        """
        Args:
            manifest_path: Path to NeMo manifest file
            vocab_path: Path to save/load vocabulary (optional)
            vocab_size: Maximum vocabulary size (None for auto)
            max_audio_len: Maximum audio length in seconds
            use_spec_augment: Whether to apply SpecAugment
            spec_augment_config: SpecAugment configuration dict
        """
        self.manifest_path = manifest_path
        self.vocab_path = vocab_path
        self.max_vocab_size = vocab_size  # Noneまたは最大値
        self.max_audio_len = max_audio_len
        self.sample_rate = 16000
        self.use_spec_augment = use_spec_augment

        # Load manifest
        self.data = self._load_manifest(manifest_path)

        # Build or load character vocabulary
        if vocab_path and Path(vocab_path).exists():
            logger.info(f"Loading vocabulary from {vocab_path}")
            self.char_to_idx, self.idx_to_char = self._load_vocab(vocab_path)
        else:
            logger.info("Building vocabulary from data")
            self.char_to_idx = self._build_vocab()
            self.idx_to_char = {v: k for k, v in self.char_to_idx.items()}

            # Save vocabulary if path is provided
            if vocab_path:
                self._save_vocab(vocab_path)
                logger.info(f"Vocabulary saved to {vocab_path}")

        self.vocab_size = len(self.char_to_idx)  # 実際のvocabサイズ

        # Setup SpecAugment if enabled
        if self.use_spec_augment and spec_augment_config:
            self.spec_augment = self._setup_spec_augment(spec_augment_config)
        else:
            self.spec_augment = None

        logger.info(f"Loaded {len(self.data)} samples from {manifest_path}")
        logger.info(f"Vocabulary size: {self.vocab_size}")
        if self.use_spec_augment:
            logger.info(f"SpecAugment enabled: {spec_augment_config}")

    def _load_manifest(self, manifest_path: str) -> List[Dict]:
        """Load NeMo manifest file (JSON Lines format)"""
        data = []
        with open(manifest_path, "r", encoding="utf-8") as f:
            for line in f:
                item = json.loads(line.strip())
                if item.get("duration", 0) <= self.max_audio_len:
                    data.append(item)
        return data

    def _build_vocab(self) -> Dict[str, int]:
        """
        Build character-level vocabulary from manifest
        Returns mapping from character to index
        Automatically determines vocab size from data
        """
        # Special tokens
        vocab = {
            "<blank>": 0,  # CTC blank token
            "<unk>": 1,
            "<sos>": 2,
            "<eos>": 3,
        }

        # Collect all unique characters
        chars = set()
        for item in self.data:
            text = item["text"]
            chars.update(text)

        # Add characters to vocab (sorted for reproducibility)
        for char in sorted(chars):
            if char not in vocab:
                vocab[char] = len(vocab)

        # Check if max_vocab_size constraint is violated
        if self.max_vocab_size is not None and len(vocab) > self.max_vocab_size:
            logger.warning(
                f"Actual vocabulary size {len(vocab)} exceeds max_vocab_size {self.max_vocab_size}. "
                f"Keeping top {self.max_vocab_size} most frequent characters."
            )
            # Count character frequencies
            char_freq = {}
            for item in self.data:
                for char in item["text"]:
                    char_freq[char] = char_freq.get(char, 0) + 1

            # Keep special tokens + most frequent characters
            special_tokens = ["<blank>", "<unk>", "<sos>", "<eos>"]
            sorted_chars = sorted(char_freq.items(), key=lambda x: x[1], reverse=True)

            vocab = {token: i for i, token in enumerate(special_tokens)}
            remaining_slots = self.max_vocab_size - len(special_tokens)

            for char, _ in sorted_chars[:remaining_slots]:
                if char not in vocab:
                    vocab[char] = len(vocab)

        return vocab

    def _save_vocab(self, vocab_path: str):
        """Save vocabulary to file"""
        vocab_data = {
            "char_to_idx": self.char_to_idx,
            "idx_to_char": self.idx_to_char,
            "vocab_size": len(self.char_to_idx),
        }
        with open(vocab_path, "wb") as f:
            pickle.dump(vocab_data, f)

    def _load_vocab(self, vocab_path: str):
        """Load vocabulary from file"""
        with open(vocab_path, "rb") as f:
            vocab_data = pickle.load(f)
        return vocab_data["char_to_idx"], vocab_data["idx_to_char"]

    def _setup_spec_augment(self, config: Dict):
        """Setup SpecAugment transformations"""
        return {
            "time_mask_num": config.get("time_mask_num", 2),
            "time_mask_width": config.get("time_mask_width", 100),
            "freq_mask_num": config.get("freq_mask_num", 2),
            "freq_mask_width": config.get("freq_mask_width", 27),
        }

    def _apply_spec_augment(self, waveform: torch.Tensor) -> torch.Tensor:
        """Apply time masking augmentation to waveform"""
        if self.spec_augment is None:
            return waveform

        time_mask_width = self.spec_augment["time_mask_width"]
        time_mask_num = self.spec_augment["time_mask_num"]

        augmented = waveform.clone()
        for _ in range(time_mask_num):
            if len(augmented) > time_mask_width:
                start = torch.randint(0, len(augmented) - time_mask_width, (1,)).item()
                augmented[start : start + time_mask_width] = 0

        return augmented

    def _encode_text(self, text: str) -> List[int]:
        """Encode text to label IDs"""
        return [self.char_to_idx.get(char, self.char_to_idx["<unk>"]) for char in text]

    def _decode_text(self, label_ids: List[int]) -> str:
        """Decode label IDs to text"""
        return "".join([self.idx_to_char.get(idx, "<unk>") for idx in label_ids])

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a single sample"""
        item = self.data[idx]

        # Load audio
        audio_path = item["audio_filepath"]
        waveform, sr = torchaudio.load(audio_path)

        # Ensure mono and correct sample rate
        if waveform.size(0) > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        waveform = waveform.squeeze(0)

        # Apply SpecAugment if enabled
        if self.use_spec_augment:
            waveform = self._apply_spec_augment(waveform)

        # Encode text to labels
        text = item["text"]
        labels = self._encode_text(text)

        return {
            "wav": waveform,
            "wav_len": len(waveform),
            "label": torch.tensor(labels, dtype=torch.long),
            "label_len": len(labels),
            "text": text,
        }

    @staticmethod
    def collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Collate function for DataLoader"""
        wavs = [item["wav"] for item in batch]
        wav_lens = torch.tensor([item["wav_len"] for item in batch], dtype=torch.long)
        labels = [item["label"] for item in batch]
        label_lens = torch.tensor([item["label_len"] for item in batch], dtype=torch.long)
        texts = [item["text"] for item in batch]

        wavs_padded = pad_sequence(wavs, batch_first=True, padding_value=0.0)
        labels_padded = pad_sequence(labels, batch_first=True, padding_value=-100)

        max_len = wavs_padded.size(1)
        attention_mask = torch.arange(max_len).unsqueeze(0) < wav_lens.unsqueeze(1)

        return {
            "wav": wavs_padded,
            "wav_len": wav_lens,
            "label": labels_padded,
            "label_len": label_lens,
            "attention_mask": attention_mask.float(),
            "text": texts,
        }
