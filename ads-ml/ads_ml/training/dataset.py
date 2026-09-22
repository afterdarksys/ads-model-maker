"""
ADS Dataset classes for training.
"""

from typing import Optional
import torch
from torch.utils.data import Dataset

# Padding id 0 is not a content token. Callers that build labels for
# generation must ignore it or the model learns to predict padding.
PAD_ID = 0


def encode_text(text: str, max_length: int, vocab_size: int = 32000) -> tuple[torch.Tensor, torch.Tensor]:
    """Character encode text. The attention mask is 0 on padding positions."""
    limit = vocab_size - 1
    ids = [min(ord(c), limit) for c in text[:max_length]]
    # A literal NUL would collide with PAD_ID. Shift it so padding stays distinguishable.
    ids = [1 if token == PAD_ID else token for token in ids]
    attention = [1] * len(ids)
    pad = max_length - len(ids)
    if pad:
        ids.extend([PAD_ID] * pad)
        attention.extend([0] * pad)
    return (
        torch.tensor(ids, dtype=torch.long),
        torch.tensor(attention, dtype=torch.long),
    )


def labels_for_generation(input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    """Copy token ids and set padding positions to the loss ignore index."""
    labels = input_ids.clone()
    labels = labels.masked_fill(attention_mask == 0, -100)
    return labels


class ADSDataset(Dataset):
    """Base dataset class."""

    def __len__(self) -> int:
        raise NotImplementedError

    def __getitem__(self, idx: int) -> dict:
        raise NotImplementedError


class ClassificationDataset(ADSDataset):
    """Dataset for text classification."""

    def __init__(
        self,
        texts: list[str],
        labels: list[str],
        label_names: list[str],
        tokenizer = None,
        max_length: int = 512,
    ):
        self.texts = texts
        self.labels = labels
        self.label_names = label_names
        self.label_to_id = {name: i for i, name in enumerate(label_names)}
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]
        label = self.labels[idx]
        try:
            label_id = self.label_to_id[label]
        except KeyError as exc:
            known = ", ".join(self.label_names)
            raise KeyError(f"unknown label {label!r}; expected one of: {known}") from exc

        if self.tokenizer:
            # Use tokenizer
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
            return {
                'input_ids': encoding['input_ids'].squeeze(0),
                'attention_mask': encoding['attention_mask'].squeeze(0),
                'labels': torch.tensor(label_id),
            }
        else:
            input_ids, attention_mask = encode_text(text, self.max_length)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': torch.tensor(label_id),
            }


class EmbeddingDataset(ADSDataset):
    """Dataset for training embeddings with contrastive learning."""

    def __init__(
        self,
        anchor_texts: list[str],
        positive_texts: list[str],
        tokenizer = None,
        max_length: int = 512,
    ):
        assert len(anchor_texts) == len(positive_texts)
        self.anchor_texts = anchor_texts
        self.positive_texts = positive_texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.anchor_texts)

    def __getitem__(self, idx: int) -> dict:
        anchor = self.anchor_texts[idx]
        positive = self.positive_texts[idx]

        if self.tokenizer:
            anchor_enc = self.tokenizer(
                anchor,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
            positive_enc = self.tokenizer(
                positive,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
            return {
                'anchor_ids': anchor_enc['input_ids'].squeeze(0),
                'anchor_mask': anchor_enc['attention_mask'].squeeze(0),
                'positive_ids': positive_enc['input_ids'].squeeze(0),
                'positive_mask': positive_enc['attention_mask'].squeeze(0),
            }
        else:
            anchor_ids, anchor_mask = encode_text(anchor, self.max_length)
            positive_ids, positive_mask = encode_text(positive, self.max_length)
            return {
                'anchor_ids': anchor_ids,
                'anchor_mask': anchor_mask,
                'positive_ids': positive_ids,
                'positive_mask': positive_mask,
            }


class GenerationDataset(ADSDataset):
    """Dataset for autoregressive text generation."""

    def __init__(
        self,
        texts: list[str],
        tokenizer = None,
        max_length: int = 1024,
    ):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> dict:
        text = self.texts[idx]

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels_for_generation(input_ids, attention_mask),
            }
        else:
            input_ids, attention_mask = encode_text(text, self.max_length)
            return {
                'input_ids': input_ids,
                'attention_mask': attention_mask,
                'labels': labels_for_generation(input_ids, attention_mask),
            }


class ChunkDataset(ADSDataset):
    """Dataset built from ingested document chunks."""

    def __init__(
        self,
        chunks: list,  # List of Chunk objects
        tokenizer = None,
        max_length: int = 512,
        task: str = 'generation',  # generation, embedding
    ):
        self.chunks = chunks
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.task = task

    def __len__(self) -> int:
        return len(self.chunks)

    def __getitem__(self, idx: int) -> dict:
        chunk = self.chunks[idx]
        text = chunk.text

        if self.tokenizer:
            encoding = self.tokenizer(
                text,
                truncation=True,
                max_length=self.max_length,
                padding='max_length',
                return_tensors='pt',
            )
            input_ids = encoding['input_ids'].squeeze(0)
            attention_mask = encoding['attention_mask'].squeeze(0)
        else:
            input_ids, attention_mask = encode_text(text, self.max_length)

        result = {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
        }

        if self.task == 'generation':
            result['labels'] = labels_for_generation(input_ids, attention_mask)

        return result
