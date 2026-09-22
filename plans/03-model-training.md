# Model Training Workflow

## Overview

The training system takes processed data and produces fine-tuned models. We leverage Hugging Face infrastructure while hiding complexity from users. Models are trained on our infrastructure and delivered ready-to-use.

## Training Philosophy

1. **Sensible Defaults**: Users shouldn't need to understand hyperparameters
2. **Transparent Costs**: Show estimated cost before training starts
3. **Interruptible**: Users can pause/resume training
4. **Observable**: Clear progress indicators and metrics
5. **Reproducible**: Same data + config = same model

## Supported Model Types

### Text Models

| Type | Base Model | Use Case |
|------|------------|----------|
| Classification | `distilbert-base-uncased` | Categorize documents, tickets, emails |
| Q&A | `distilbert-base-uncased-distilled-squad` | Answer questions from documents |
| Generation | `gpt2`, `opt-350m` | Generate text similar to training data |
| Embedding | `sentence-transformers/all-MiniLM-L6-v2` | Semantic search, similarity |
| NER | `bert-base-ner` | Entity extraction |
| Summarization | `facebook/bart-large-cnn` | Document summarization |

### Vision Models

| Type | Base Model | Use Case |
|------|------------|----------|
| Classification | `google/vit-base-patch16-224` | Image categorization |
| Object Detection | `facebook/detr-resnet-50` | Detect objects in images |
| OCR Enhancement | `microsoft/trocr-base-printed` | Better text extraction |
| Document Layout | `microsoft/layoutlmv3-base` | Understand document structure |

### Multimodal Models

| Type | Base Model | Use Case |
|------|------------|----------|
| Document Q&A | `microsoft/layoutlmv3-base` | Q&A on documents with images |
| Image Captioning | `Salesforce/blip-image-captioning-base` | Describe images |

## Training Pipeline

```
┌─────────────────────────────────────────────────────────────────┐
│                     TRAINING PIPELINE                            │
└─────────────────────────────────────────────────────────────────┘

┌──────────────┐
│  User Config │
│  - Model type│
│  - Size pref │
│  - Quality   │
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│  Config Resolver     │  ← Map user preferences to training params
│  - Select base model │
│  - Set hyperparams   │
│  - Estimate cost     │
└──────────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Data Loader         │
│  - Load chunks       │
│  - Create datasets   │
│  - Train/val split   │
└──────────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Model Initialization│
│  - Load base model   │
│  - Configure LoRA    │
│  - Setup optimizer   │
└──────────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Training Loop       │
│  - Forward pass      │
│  - Loss computation  │
│  - Backward pass     │
│  - Checkpoint save   │
└──────────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Evaluation          │
│  - Validation metrics│
│  - Benchmark tests   │
│  - Quality score     │
└──────────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Export              │
│  - Merge LoRA weights│
│  - Quantize (opt)    │
│  - Save formats      │
└──────────┬───────────┘
       │
       ▼
┌──────────────────────┐
│  Publish             │
│  - Upload to HF      │
│  - Store in cloud    │
│  - Notify user       │
└──────────────────────┘
```

## Parameter-Efficient Fine-Tuning (PEFT)

We use LoRA (Low-Rank Adaptation) to reduce training costs:

```python
from peft import LoraConfig, get_peft_model, TaskType

class LoRATrainer:
    """
    Fine-tune models using LoRA for efficiency.
    Reduces GPU memory requirements by 10-100x.
    """

    DEFAULT_CONFIGS = {
        'small': LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=8,
            lora_alpha=16,
            lora_dropout=0.1,
            target_modules=['query', 'value'],
        ),
        'medium': LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=16,
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=['query', 'key', 'value', 'output'],
        ),
        'large': LoraConfig(
            task_type=TaskType.SEQ_CLS,
            r=32,
            lora_alpha=64,
            lora_dropout=0.05,
            target_modules=['query', 'key', 'value', 'output', 'dense'],
        ),
    }

    def __init__(self, model_name: str, config_size: str = 'medium'):
        self.base_model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=self.num_labels,
        )
        self.lora_config = self.DEFAULT_CONFIGS[config_size]
        self.model = get_peft_model(self.base_model, self.lora_config)

        # Print trainable parameters
        self.model.print_trainable_parameters()
        # Output: trainable params: 294,912 || all params: 66,955,010 || trainable%: 0.44
```

## Training Configurations

### User-Friendly Presets

```python
TRAINING_PRESETS = {
    'quick': {
        'description': 'Fast training, good for testing',
        'epochs': 1,
        'batch_size': 32,
        'learning_rate': 5e-4,
        'lora_config': 'small',
        'estimated_time_factor': 1.0,
    },
    'balanced': {
        'description': 'Good balance of speed and quality',
        'epochs': 3,
        'batch_size': 16,
        'learning_rate': 2e-4,
        'lora_config': 'medium',
        'estimated_time_factor': 3.0,
    },
    'thorough': {
        'description': 'Best quality, longer training',
        'epochs': 5,
        'batch_size': 8,
        'learning_rate': 1e-4,
        'lora_config': 'large',
        'warmup_ratio': 0.1,
        'estimated_time_factor': 6.0,
    },
    'maximum': {
        'description': 'Maximum quality for production',
        'epochs': 10,
        'batch_size': 4,
        'learning_rate': 5e-5,
        'lora_config': 'large',
        'warmup_ratio': 0.1,
        'gradient_accumulation_steps': 4,
        'estimated_time_factor': 15.0,
    },
}
```

### Model Size Profiles

```python
MODEL_SIZE_PROFILES = {
    'tiny': {
        'description': 'Minimal footprint, edge deployment',
        'base_models': {
            'classification': 'prajjwal1/bert-tiny',
            'qa': 'distilbert-base-uncased-distilled-squad',
            'embedding': 'sentence-transformers/all-MiniLM-L6-v2',
        },
        'max_params': '50M',
        'ram_estimate': '200MB',
        'disk_estimate': '100MB',
    },
    'small': {
        'description': 'Good for most use cases',
        'base_models': {
            'classification': 'distilbert-base-uncased',
            'qa': 'distilbert-base-uncased-distilled-squad',
            'embedding': 'sentence-transformers/all-mpnet-base-v2',
        },
        'max_params': '150M',
        'ram_estimate': '600MB',
        'disk_estimate': '300MB',
    },
    'medium': {
        'description': 'Higher accuracy',
        'base_models': {
            'classification': 'bert-base-uncased',
            'qa': 'bert-large-uncased-whole-word-masking-finetuned-squad',
            'generation': 'gpt2-medium',
        },
        'max_params': '500M',
        'ram_estimate': '2GB',
        'disk_estimate': '1GB',
    },
    'large': {
        'description': 'Best accuracy, server deployment',
        'base_models': {
            'classification': 'roberta-large',
            'qa': 'deepset/roberta-large-squad2',
            'generation': 'gpt2-large',
        },
        'max_params': '1B',
        'ram_estimate': '4GB',
        'disk_estimate': '2GB',
    },
}
```

## Training Orchestrator

```python
class TrainingOrchestrator:
    """
    Orchestrate the full training pipeline.
    Handles job lifecycle, checkpointing, and reporting.
    """

    def __init__(self, config: TrainingConfig):
        self.config = config
        self.job_id = config.job_id
        self.storage = StorageClient()
        self.metrics = MetricsReporter(self.job_id)

    async def run(self) -> TrainingResult:
        """Execute training pipeline"""

        try:
            # Update job status
            await self.update_status('initializing')

            # Load processed data
            dataset = await self.load_dataset()
            self.metrics.report('dataset_loaded', {
                'train_samples': len(dataset['train']),
                'val_samples': len(dataset['validation']),
            })

            # Initialize model
            model, tokenizer = await self.initialize_model()
            await self.update_status('training')

            # Setup trainer
            trainer = self.create_trainer(model, tokenizer, dataset)

            # Training loop with callbacks
            train_result = trainer.train()

            # Evaluation
            await self.update_status('evaluating')
            eval_result = trainer.evaluate()
            self.metrics.report('evaluation', eval_result)

            # Export model
            await self.update_status('exporting')
            export_path = await self.export_model(trainer, tokenizer)

            # Upload to storage
            await self.update_status('uploading')
            model_url = await self.upload_model(export_path)

            # Complete
            await self.update_status('completed')

            return TrainingResult(
                job_id=self.job_id,
                model_url=model_url,
                metrics=eval_result,
                training_time=train_result.metrics['train_runtime'],
            )

        except Exception as e:
            await self.update_status('failed', error=str(e))
            raise

    def create_trainer(
        self,
        model,
        tokenizer,
        dataset,
    ) -> Trainer:
        """Create HuggingFace Trainer with our configuration"""

        training_args = TrainingArguments(
            output_dir=f'/tmp/training/{self.job_id}',
            num_train_epochs=self.config.epochs,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size * 2,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            weight_decay=0.01,
            logging_dir=f'/tmp/logs/{self.job_id}',
            logging_steps=10,
            evaluation_strategy='steps',
            eval_steps=100,
            save_strategy='steps',
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model='eval_loss',
            greater_is_better=False,
            fp16=torch.cuda.is_available(),
            report_to=['tensorboard'],
        )

        return Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset['train'],
            eval_dataset=dataset['validation'],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer),
            callbacks=[
                ProgressCallback(self.job_id),
                CheckpointCallback(self.storage),
                EarlyStoppingCallback(patience=3),
            ],
        )
```

## Cost Estimation

```python
class CostEstimator:
    """
    Estimate training costs based on data size and configuration.
    """

    # Base costs per GPU hour (A100 40GB)
    GPU_HOUR_COST = 2.50

    # Tokens processed per second estimates
    TOKENS_PER_SECOND = {
        'tiny': 50000,
        'small': 30000,
        'medium': 15000,
        'large': 8000,
    }

    def estimate(
        self,
        dataset_stats: DatasetStats,
        config: TrainingConfig,
    ) -> CostEstimate:
        """Calculate estimated training cost"""

        # Calculate total tokens
        total_tokens = dataset_stats.total_tokens

        # Account for epochs
        total_tokens *= config.epochs

        # Get processing speed
        tokens_per_second = self.TOKENS_PER_SECOND[config.model_size]

        # Apply batch size factor
        tokens_per_second *= (config.batch_size / 16)

        # Calculate GPU hours
        gpu_seconds = total_tokens / tokens_per_second
        gpu_hours = gpu_seconds / 3600

        # Add overhead (data loading, evaluation, export)
        gpu_hours *= 1.2

        # Calculate cost
        cost = gpu_hours * self.GPU_HOUR_COST

        # Apply plan discount
        cost *= self.get_plan_discount(config.plan_tier)

        return CostEstimate(
            gpu_hours=gpu_hours,
            estimated_cost=cost,
            estimated_time_minutes=int(gpu_hours * 60),
            confidence='medium',
            breakdown={
                'training': cost * 0.7,
                'evaluation': cost * 0.1,
                'storage': cost * 0.1,
                'overhead': cost * 0.1,
            }
        )
```

## Checkpointing & Resume

```python
class CheckpointManager:
    """
    Manage training checkpoints for resume capability.
    """

    def __init__(self, job_id: str, storage: StorageClient):
        self.job_id = job_id
        self.storage = storage
        self.checkpoint_path = f'checkpoints/{job_id}'

    async def save_checkpoint(
        self,
        trainer: Trainer,
        step: int,
        metrics: dict,
    ) -> str:
        """Save checkpoint to cloud storage"""

        local_path = f'/tmp/checkpoints/{self.job_id}/step-{step}'
        trainer.save_model(local_path)

        # Upload to cloud
        remote_path = f'{self.checkpoint_path}/step-{step}'
        await self.storage.upload_directory(local_path, remote_path)

        # Save metadata
        metadata = {
            'step': step,
            'metrics': metrics,
            'timestamp': datetime.utcnow().isoformat(),
        }
        await self.storage.upload_json(
            f'{remote_path}/metadata.json',
            metadata,
        )

        return remote_path

    async def load_checkpoint(self) -> Optional[str]:
        """Load latest checkpoint for resume"""

        checkpoints = await self.storage.list_directory(self.checkpoint_path)
        if not checkpoints:
            return None

        # Find latest checkpoint
        latest = max(checkpoints, key=lambda c: c['step'])
        local_path = f'/tmp/checkpoints/{self.job_id}/resume'

        await self.storage.download_directory(latest['path'], local_path)
        return local_path
```

## Model Export

```python
class ModelExporter:
    """
    Export trained models to various formats.
    """

    async def export(
        self,
        trainer: Trainer,
        tokenizer,
        formats: list[str],
    ) -> dict[str, str]:
        """Export model to specified formats"""

        exports = {}
        base_path = f'/tmp/exports/{trainer.args.output_dir}'

        # Merge LoRA weights if applicable
        if hasattr(trainer.model, 'merge_and_unload'):
            merged_model = trainer.model.merge_and_unload()
        else:
            merged_model = trainer.model

        for format_type in formats:
            if format_type == 'pytorch':
                path = await self._export_pytorch(merged_model, tokenizer, base_path)
            elif format_type == 'safetensors':
                path = await self._export_safetensors(merged_model, tokenizer, base_path)
            elif format_type == 'onnx':
                path = await self._export_onnx(merged_model, tokenizer, base_path)
            elif format_type == 'ggml':
                path = await self._export_ggml(merged_model, base_path)
            elif format_type == 'coreml':
                path = await self._export_coreml(merged_model, base_path)
            else:
                continue

            exports[format_type] = path

        return exports

    async def _export_safetensors(
        self,
        model,
        tokenizer,
        base_path: str,
    ) -> str:
        """Export in safetensors format (default)"""

        export_path = f'{base_path}/safetensors'
        os.makedirs(export_path, exist_ok=True)

        # Save model
        model.save_pretrained(export_path, safe_serialization=True)
        tokenizer.save_pretrained(export_path)

        return export_path

    async def _export_ggml(
        self,
        model,
        base_path: str,
    ) -> str:
        """Export for llama.cpp / local inference"""

        export_path = f'{base_path}/ggml'
        os.makedirs(export_path, exist_ok=True)

        # Use llama.cpp conversion script
        # This enables running on CPU without Python
        subprocess.run([
            'python', '-m', 'llama_cpp.convert',
            '--outfile', f'{export_path}/model.gguf',
            '--outtype', 'f16',
            model.config._name_or_path,
        ])

        return export_path
```

## Hugging Face Integration

```python
class HuggingFaceManager:
    """
    Manage models on Hugging Face Hub.
    Models are stored in private repos, invisible to users.
    """

    def __init__(self, token: str, org: str = 'ads-model-maker'):
        self.api = HfApi(token=token)
        self.org = org

    async def create_repo(self, model_id: str, user_id: str) -> str:
        """Create private repo for model"""

        repo_name = f'{self.org}/{user_id}-{model_id}'

        self.api.create_repo(
            repo_id=repo_name,
            private=True,
            repo_type='model',
        )

        return repo_name

    async def upload_model(
        self,
        local_path: str,
        repo_name: str,
        commit_message: str = 'Upload trained model',
    ) -> str:
        """Upload model to HF Hub"""

        self.api.upload_folder(
            folder_path=local_path,
            repo_id=repo_name,
            commit_message=commit_message,
        )

        return f'https://huggingface.co/{repo_name}'

    async def delete_model(self, repo_name: str) -> None:
        """Delete model repo (when user deletes model)"""
        self.api.delete_repo(repo_id=repo_name)
```

## Training Metrics & Observability

```python
@dataclass
class TrainingMetrics:
    """Metrics collected during training"""

    # Training progress
    current_epoch: int
    current_step: int
    total_steps: int
    progress_percent: float

    # Loss metrics
    train_loss: float
    eval_loss: float
    best_loss: float

    # Performance metrics (task-specific)
    accuracy: Optional[float]
    f1_score: Optional[float]
    precision: Optional[float]
    recall: Optional[float]

    # Resource usage
    gpu_memory_used: int
    gpu_utilization: float
    training_time_seconds: int

    # Estimates
    estimated_time_remaining: int
    estimated_cost_remaining: float
```

## Quality Assurance

```python
class ModelQualityChecker:
    """
    Validate model quality before delivery.
    """

    def __init__(self, model, tokenizer, test_data: Dataset):
        self.model = model
        self.tokenizer = tokenizer
        self.test_data = test_data

    async def run_quality_checks(self) -> QualityReport:
        """Run comprehensive quality checks"""

        results = {}

        # Basic inference test
        results['inference_test'] = await self._test_inference()

        # Performance benchmarks
        results['latency'] = await self._measure_latency()
        results['throughput'] = await self._measure_throughput()

        # Accuracy on held-out test set
        results['test_metrics'] = await self._evaluate_test_set()

        # Edge case testing
        results['edge_cases'] = await self._test_edge_cases()

        # Memory profiling
        results['memory_profile'] = await self._profile_memory()

        return QualityReport(
            passed=all(r['passed'] for r in results.values()),
            results=results,
            recommendations=self._generate_recommendations(results),
        )
```

## Configuration Schema

```json
{
  "training": {
    "job_id": "uuid",
    "user_id": "uuid",
    "data_source_ids": ["uuid"],
    "model_type": "classification|qa|generation|embedding",
    "model_size": "tiny|small|medium|large",
    "quality_preset": "quick|balanced|thorough|maximum",
    "custom_config": {
      "epochs": 3,
      "batch_size": 16,
      "learning_rate": 2e-4,
      "warmup_ratio": 0.1
    },
    "export_formats": ["safetensors", "onnx", "ggml"],
    "notifications": {
      "email_on_complete": true,
      "webhook_url": "https://..."
    }
  }
}
```
