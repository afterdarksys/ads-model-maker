"""
ADS Classifier - Custom text classification model.

Builds on ADSEncoder with classification head.
Supports multi-class and multi-label classification.
"""

from typing import Optional, Literal
from dataclasses import dataclass
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from ads_ml.models.blocks import ADSEncoder


@dataclass
class ClassifierConfig:
    """Configuration for ADSClassifier."""
    vocab_size: int = 32000
    num_labels: int = 2
    preset: str = 'small'
    dim: int = 384
    depth: int = 6
    num_heads: int = 6
    max_seq_len: int = 512
    dropout: float = 0.1
    classifier_dropout: float = 0.1
    problem_type: Literal['single_label', 'multi_label'] = 'single_label'
    label_names: Optional[list[str]] = None

    def to_dict(self) -> dict:
        return {
            'vocab_size': self.vocab_size,
            'num_labels': self.num_labels,
            'preset': self.preset,
            'dim': self.dim,
            'depth': self.depth,
            'num_heads': self.num_heads,
            'max_seq_len': self.max_seq_len,
            'dropout': self.dropout,
            'classifier_dropout': self.classifier_dropout,
            'problem_type': self.problem_type,
            'label_names': self.label_names,
        }

    @classmethod
    def from_dict(cls, d: dict) -> 'ClassifierConfig':
        return cls(**d)

    def save(self, path: Path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> 'ClassifierConfig':
        with open(path) as f:
            return cls.from_dict(json.load(f))


class ADSClassifier(nn.Module):
    """
    Custom text classifier with our own architecture.

    Features:
    - Multiple model size presets (micro to large)
    - Single-label and multi-label classification
    - Confidence scores and probabilities
    - ONNX export ready
    - GoLearn export ready
    """

    def __init__(self, config: ClassifierConfig):
        super().__init__()
        self.config = config

        # Build encoder
        self.encoder = ADSEncoder.from_preset(
            preset=config.preset,
            vocab_size=config.vocab_size,
            dim=config.dim,
            depth=config.depth,
            num_heads=config.num_heads,
            max_seq_len=config.max_seq_len,
            dropout=config.dropout,
        )

        # Classification head
        self.classifier_dropout = nn.Dropout(config.classifier_dropout)
        self.classifier = nn.Linear(self.encoder.dim, config.num_labels)

        # Label names for inference
        self.label_names = config.label_names or [
            f'label_{i}' for i in range(config.num_labels)
        ]

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        """
        Forward pass with optional loss computation.

        Args:
            input_ids: Token IDs [batch, seq_len]
            attention_mask: Mask for padding [batch, seq_len]
            labels: Ground truth labels for training

        Returns:
            Dictionary with logits, loss (if labels provided), predictions
        """
        # Encode
        pooled = self.encoder(input_ids, attention_mask)
        pooled = self.classifier_dropout(pooled)

        # Classify
        logits = self.classifier(pooled)

        output = {'logits': logits}

        # Compute predictions
        if self.config.problem_type == 'single_label':
            probs = F.softmax(logits, dim=-1)
            output['probabilities'] = probs
            output['predictions'] = logits.argmax(dim=-1)
            output['confidence'] = probs.max(dim=-1).values
        else:  # multi_label
            probs = torch.sigmoid(logits)
            output['probabilities'] = probs
            output['predictions'] = (probs > 0.5).long()
            output['confidence'] = probs

        # Compute loss if labels provided
        if labels is not None:
            if self.config.problem_type == 'single_label':
                loss = F.cross_entropy(logits, labels)
            else:
                loss = F.binary_cross_entropy_with_logits(logits, labels.float())
            output['loss'] = loss

        return output

    def predict(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> list[dict]:
        """
        User-friendly prediction with label names.

        Returns list of dicts with label, confidence, all_probabilities.
        """
        self.eval()
        with torch.no_grad():
            output = self.forward(input_ids, attention_mask)

        results = []
        batch_size = input_ids.shape[0]

        for i in range(batch_size):
            if self.config.problem_type == 'single_label':
                pred_idx = output['predictions'][i].item()
                results.append({
                    'label': self.label_names[pred_idx],
                    'label_id': pred_idx,
                    'confidence': output['confidence'][i].item(),
                    'probabilities': {
                        name: output['probabilities'][i, j].item()
                        for j, name in enumerate(self.label_names)
                    },
                })
            else:
                pred_labels = []
                for j, name in enumerate(self.label_names):
                    if output['predictions'][i, j]:
                        pred_labels.append({
                            'label': name,
                            'confidence': output['probabilities'][i, j].item(),
                        })
                results.append({
                    'labels': pred_labels,
                    'probabilities': {
                        name: output['probabilities'][i, j].item()
                        for j, name in enumerate(self.label_names)
                    },
                })

        return results

    def save(self, path: Path):
        """Save model and config."""
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save config
        self.config.save(path / 'config.json')

        # Save weights
        torch.save(self.state_dict(), path / 'model.pt')

        # Save label names
        with open(path / 'labels.json', 'w') as f:
            json.dump(self.label_names, f)

    @classmethod
    def load(cls, path: Path, device: str = 'cpu') -> 'ADSClassifier':
        """Load model from directory."""
        path = Path(path)

        config = ClassifierConfig.load(path / 'config.json')
        model = cls(config)
        model.load_state_dict(torch.load(path / 'model.pt', map_location=device))

        with open(path / 'labels.json') as f:
            model.label_names = json.load(f)

        return model

    def export_onnx(self, path: Path, seq_len: int = 128):
        """Export to ONNX format for Go agent."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Move to CPU for export (avoids device mismatch)
        device = next(self.parameters()).device
        self.cpu()
        self.eval()

        # Dummy inputs (on CPU)
        dummy_input_ids = torch.zeros(1, seq_len, dtype=torch.long)
        dummy_mask = torch.ones(1, seq_len, dtype=torch.bool)

        # forward() returns a dict. ONNX needs a single named tensor.
        class _LogitsOnly(nn.Module):
            def __init__(self, classifier: 'ADSClassifier'):
                super().__init__()
                self.classifier = classifier

            def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
                return self.classifier(input_ids, attention_mask)['logits']

        # Use legacy export mode for compatibility
        torch.onnx.export(
            _LogitsOnly(self),
            (dummy_input_ids, dummy_mask),
            str(path),
            input_names=['input_ids', 'attention_mask'],
            output_names=['logits'],
            dynamic_axes={
                'input_ids': {0: 'batch', 1: 'seq'},
                'attention_mask': {0: 'batch', 1: 'seq'},
                'logits': {0: 'batch'},
            },
            opset_version=14,
            dynamo=False,  # Use legacy export for single-file output
        )

        # Restore original device if needed
        if device.type != 'cpu':
            self.to(device)

    def get_num_params(self) -> int:
        """Total parameters."""
        return sum(p.numel() for p in self.parameters())

    def get_model_size_mb(self) -> float:
        """Estimated model size in MB."""
        param_bytes = sum(
            p.numel() * p.element_size() for p in self.parameters()
        )
        return param_bytes / (1024 * 1024)


def create_classifier(
    num_labels: int,
    label_names: Optional[list[str]] = None,
    size: str = 'small',
    vocab_size: int = 32000,
    problem_type: str = 'single_label',
) -> ADSClassifier:
    """
    Factory function to create a classifier.

    Args:
        num_labels: Number of classification labels
        label_names: Optional names for each label
        size: Model size preset (micro, tiny, small, base, medium, large)
        vocab_size: Vocabulary size
        problem_type: 'single_label' or 'multi_label'

    Returns:
        ADSClassifier instance
    """
    presets = ADSEncoder.CONFIGS

    if size not in presets:
        raise ValueError(f"Unknown size '{size}'. Choose from: {list(presets.keys())}")

    # Get preset and remap 'heads' to 'num_heads' for ClassifierConfig compatibility
    preset_config = presets[size].copy()
    if 'heads' in preset_config:
        preset_config['num_heads'] = preset_config.pop('heads')

    config = ClassifierConfig(
        vocab_size=vocab_size,
        num_labels=num_labels,
        preset=size,
        label_names=label_names,
        problem_type=problem_type,
        **preset_config,
    )

    return ADSClassifier(config)
