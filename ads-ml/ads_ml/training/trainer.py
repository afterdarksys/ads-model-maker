"""
ADS Trainer - Train custom models from data.
"""

import math
import random
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Callable, Iterator
import json

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR

import structlog

from ads_ml.models import ADSClassifier, ADSEmbedder, ADSGenerator
from ads_ml.training.dataset import ADSDataset

logger = structlog.get_logger()


@dataclass
class TrainingConfig:
    """Training configuration."""
    # Training params
    epochs: int = 3
    batch_size: int = 16
    learning_rate: float = 2e-4
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Model params
    model_type: str = 'classifier'  # classifier, embedder, generator
    model_size: str = 'small'

    # Data params
    max_seq_len: int = 512

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    output_dir: str = './output'

    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001

    def to_dict(self) -> dict:
        return {k: v for k, v in vars(self).items()}


@dataclass
class TrainingMetrics:
    """Metrics collected during training."""
    epoch: int = 0
    step: int = 0
    train_loss: float = 0.0
    eval_loss: float = 0.0
    eval_accuracy: float = 0.0
    learning_rate: float = 0.0
    samples_per_second: float = 0.0
    gpu_memory_mb: float = 0.0


class ADSTrainer:
    """
    Train ADS models from scratch.

    Supports:
    - Classification models
    - Embedding models
    - Generation models
    """

    def __init__(
        self,
        model: nn.Module,
        config: TrainingConfig,
        train_dataset: ADSDataset,
        eval_dataset: Optional[ADSDataset] = None,
        tokenizer = None,
    ):
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.tokenizer = tokenizer

        # Setup device
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # Setup optimizer
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

        # Integer division drops to zero when the dataset is smaller than
        # the batch, and CosineAnnealingLR rejects T_max < 1.
        steps_per_epoch = max(1, math.ceil(len(train_dataset) / max(config.batch_size, 1)))
        total_steps = max(1, steps_per_epoch * max(config.epochs, 1))
        self.scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=total_steps,
        )

        # Setup output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.global_step = 0
        self.best_eval_loss = float('inf')
        self.patience_counter = 0
        self.history: list[TrainingMetrics] = []

        # Callbacks
        self.callbacks: list[Callable] = []

    def add_callback(self, callback: Callable):
        """Add a training callback."""
        self.callbacks.append(callback)

    def train(self) -> dict:
        """Run the full training loop."""
        if len(self.train_dataset) == 0:
            raise ValueError("training dataset is empty")
        if self.config.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if self.config.epochs < 1:
            raise ValueError("epochs must be at least 1")

        logger.info(
            "starting_training",
            epochs=self.config.epochs,
            batch_size=self.config.batch_size,
            device=str(self.device),
        )

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
            pin_memory=True if self.device.type == 'cuda' else False,
        )

        start_time = time.time()

        for epoch in range(self.config.epochs):
            epoch_loss = self._train_epoch(train_loader, epoch)

            # Evaluate
            if self.eval_dataset:
                eval_metrics = self.evaluate()
                logger.info(
                    "epoch_complete",
                    epoch=epoch + 1,
                    train_loss=epoch_loss,
                    eval_loss=eval_metrics['loss'],
                    eval_accuracy=eval_metrics.get('accuracy', 0),
                    eval_macro_f1=eval_metrics.get('macro_f1', 0),
                )

                # Early stopping check
                if self._check_early_stopping(eval_metrics['loss']):
                    logger.info("early_stopping_triggered")
                    break
            else:
                logger.info("epoch_complete", epoch=epoch + 1, train_loss=epoch_loss)

            # Save checkpoint
            self._save_checkpoint(epoch)

        total_time = time.time() - start_time

        # Save final model
        self._save_final()

        return {
            'total_steps': self.global_step,
            'total_time_seconds': total_time,
            'final_train_loss': epoch_loss,
            'final_eval_loss': self.best_eval_loss,
            'history': [m.__dict__ for m in self.history],
        }

    def _train_epoch(self, loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0

        for batch in loader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }

            # Forward pass
            self.optimizer.zero_grad()
            outputs = self._forward_loss(batch)
            loss = outputs['loss']

            # Backward pass
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.max_grad_norm,
            )

            self.optimizer.step()
            self.scheduler.step()

            total_loss += loss.item()
            num_batches += 1
            self.global_step += 1

            # Logging
            if self.global_step % 10 == 0:
                metrics = TrainingMetrics(
                    epoch=epoch,
                    step=self.global_step,
                    train_loss=loss.item(),
                    learning_rate=self.scheduler.get_last_lr()[0],
                )
                self.history.append(metrics)

                # Run callbacks
                for callback in self.callbacks:
                    callback(metrics)

            # Eval during training
            if self.eval_dataset and self.global_step % self.config.eval_steps == 0:
                eval_metrics = self.evaluate()
                logger.debug(
                    "eval_step",
                    step=self.global_step,
                    loss=eval_metrics['loss'],
                )
                self.model.train()

            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self._save_checkpoint(epoch)

        if num_batches == 0:
            return 0.0
        return total_loss / num_batches

    def _forward_loss(self, batch: dict) -> dict:
        """Run the model and always return a dict that contains loss.

        Contrastive embedder batches use anchor/positive ids. Classifier and
        generator batches pass through to the module, which must return loss.
        """
        if 'anchor_ids' in batch:
            if not hasattr(self.model, 'contrastive_loss'):
                raise RuntimeError("model does not support contrastive batches")
            anchor = self.model(batch['anchor_ids'], batch.get('anchor_mask'))
            positive = self.model(batch['positive_ids'], batch.get('positive_mask'))
            loss = self.model.contrastive_loss(anchor, positive)
            return {'loss': loss}

        outputs = self.model(**batch)
        if 'loss' not in outputs:
            raise RuntimeError(
                "model forward did not return loss; include labels or use a contrastive batch"
            )
        return outputs

    @torch.no_grad()
    def evaluate(self) -> dict:
        """Evaluate on eval dataset."""
        if not self.eval_dataset:
            return {}

        self.model.eval()

        eval_loader = DataLoader(
            self.eval_dataset,
            batch_size=self.config.batch_size * 2,
            shuffle=False,
        )

        total_loss = 0.0
        total_correct = 0
        total_samples = 0
        pred_ids: list[int] = []
        label_ids: list[int] = []

        if len(eval_loader) == 0:
            return {'loss': 0.0, 'accuracy': 0.0, 'macro_f1': 0.0}

        for batch in eval_loader:
            batch = {
                k: v.to(self.device) if torch.is_tensor(v) else v
                for k, v in batch.items()
            }
            outputs = self._forward_loss(batch)

            total_loss += outputs['loss'].item()

            predictions = outputs.get('predictions')
            labels = batch.get('labels')
            # Class ids only. Token-level generation labels are sequences and
            # are not comparable to a single class prediction.
            if (
                predictions is not None
                and labels is not None
                and predictions.shape == labels.shape
                and predictions.ndim == 1
            ):
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
                pred_ids.extend(int(v) for v in predictions.detach().cpu().tolist())
                label_ids.extend(int(v) for v in labels.detach().cpu().tolist())

        avg_loss = total_loss / len(eval_loader)
        accuracy = total_correct / total_samples if total_samples > 0 else 0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'macro_f1': _macro_f1(label_ids, pred_ids),
        }

    def _check_early_stopping(self, eval_loss: float) -> bool:
        """Check if training should stop early."""
        if eval_loss < self.best_eval_loss - self.config.early_stopping_threshold:
            self.best_eval_loss = eval_loss
            self.patience_counter = 0
            return False

        self.patience_counter += 1
        return self.patience_counter >= self.config.early_stopping_patience

    def _save_checkpoint(self, epoch: int):
        """Save training checkpoint."""
        checkpoint_dir = self.output_dir / f'checkpoint-{self.global_step}'
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # Save model
        if hasattr(self.model, 'save'):
            self.model.save(checkpoint_dir)
        else:
            torch.save(self.model.state_dict(), checkpoint_dir / 'model.pt')

        # Save training state
        torch.save({
            'epoch': epoch,
            'global_step': self.global_step,
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
            'best_eval_loss': self.best_eval_loss,
        }, checkpoint_dir / 'training_state.pt')

        logger.debug("checkpoint_saved", path=str(checkpoint_dir))

    def _save_final(self):
        """Save final trained model."""
        final_dir = self.output_dir / 'final'
        final_dir.mkdir(parents=True, exist_ok=True)

        if hasattr(self.model, 'save'):
            self.model.save(final_dir)
        else:
            torch.save(self.model.state_dict(), final_dir / 'model.pt')

        # Save config
        with open(final_dir / 'training_config.json', 'w') as f:
            json.dump(self.config.to_dict(), f, indent=2)

        # Export to ONNX
        if hasattr(self.model, 'export_onnx'):
            self.model.export_onnx(final_dir / 'model.onnx')
            logger.info("onnx_exported", path=str(final_dir / 'model.onnx'))

        logger.info("model_saved", path=str(final_dir))


def train_classifier(
    train_data: list[tuple[str, str]],  # (text, label) pairs
    labels: list[str],
    model_size: str = 'small',
    epochs: int = 3,
    output_dir: str = './output',
    tokenizer = None,
) -> ADSClassifier:
    """
    Convenience function to train a classifier.

    Args:
        train_data: List of (text, label) tuples
        labels: List of unique labels
        model_size: Model size preset
        epochs: Number of training epochs
        output_dir: Where to save the model
        tokenizer: Tokenizer to use

    Returns:
        Trained ADSClassifier
    """
    from ads_ml.models.classifier import create_classifier
    from ads_ml.training.dataset import ClassificationDataset

    # Create model
    model = create_classifier(
        num_labels=len(labels),
        label_names=labels,
        size=model_size,
    )

    if not train_data:
        raise ValueError("training data is empty")
    unknown = sorted({label for _, label in train_data if label not in labels})
    if unknown:
        raise ValueError(f"training data has labels that are not in the label list: {unknown}")

    # Create dataset
    texts = [t[0] for t in train_data]
    text_labels = [t[1] for t in train_data]
    dataset = ClassificationDataset(texts, text_labels, labels, tokenizer)

    train_dataset, eval_dataset = _split_train_eval(dataset)

    # Train
    config = TrainingConfig(
        epochs=epochs,
        model_type='classifier',
        model_size=model_size,
        output_dir=output_dir,
    )

    trainer = ADSTrainer(
        model=model,
        config=config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
    )

    trainer.train()

    return model


def _split_train_eval(dataset):
    """Hold out the last 10 percent after a fixed shuffle.

    Both sides stay non-empty when there is more than one example. A single
    example is used only for training.
    """
    count = len(dataset)
    if count == 0:
        raise ValueError("training data is empty")
    if count == 1:
        return dataset, None

    order = list(range(count))
    random.Random(0).shuffle(order)
    cut = max(1, min(count - 1, int(count * 0.9)))
    return Subset(dataset, order[:cut]), Subset(dataset, order[cut:])


def _macro_f1(labels: list[int], predictions: list[int]) -> float:
    """Unweighted mean of per-class F1. Missing classes do not contribute."""
    if not labels:
        return 0.0
    classes = sorted(set(labels) | set(predictions))
    scores = []
    for class_id in classes:
        tp = sum(1 for y, p in zip(labels, predictions) if y == class_id and p == class_id)
        fp = sum(1 for y, p in zip(labels, predictions) if y != class_id and p == class_id)
        fn = sum(1 for y, p in zip(labels, predictions) if y == class_id and p != class_id)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        if precision + recall == 0:
            scores.append(0.0)
        else:
            scores.append(2 * precision * recall / (precision + recall))
    return sum(scores) / len(scores)
