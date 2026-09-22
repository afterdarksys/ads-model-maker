# System Architecture

## Overview

Three-tier architecture separating concerns:

```
┌─────────────────────────────────────────────────────────────────┐
│                         USER LAYER                               │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐  │
│  │  Web UI     │  │  CLI Tool   │  │  Go Agent (Local)       │  │
│  │  Dashboard  │  │  ads-mm     │  │  Filesystem/API/Serving │  │
│  └──────┬──────┘  └──────┬──────┘  └────────────┬────────────┘  │
└─────────┼────────────────┼──────────────────────┼───────────────┘
          │                │                      │
          ▼                ▼                      ▼
┌─────────────────────────────────────────────────────────────────┐
│                       API GATEWAY                                │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Authentication │ Rate Limiting │ Routing │ Load Balancing  ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                      SERVICE LAYER                               │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ Data Service │ │Train Service │ │Model Service │             │
│  │   (Python)   │ │   (Python)   │ │   (Python)   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │ User Service │ │Bill Service  │ │ Job Queue    │             │
│  │    (Go)      │ │    (Go)      │ │  (Redis)     │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
          │
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    INFRASTRUCTURE LAYER                          │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │  PostgreSQL  │ │    Redis     │ │ Object Store │             │
│  │   (Users,    │ │   (Queue,    │ │   (S3/OCI)   │             │
│  │    Jobs)     │ │    Cache)    │ │              │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐             │
│  │  GPU Cluster │ │ Hugging Face │ │ Dark Storage │             │
│  │  (Training)  │ │   (Models)   │ │  (Trained)   │             │
│  └──────────────┘ └──────────────┘ └──────────────┘             │
└─────────────────────────────────────────────────────────────────┘
```

## Component Details

### 1. Go Agent (User's Machine)

```
ads-agent
├── cmd/
│   └── agent/
│       └── main.go           # Entry point
├── internal/
│   ├── api/
│   │   ├── server.go         # Local HTTP server
│   │   ├── handlers.go       # API endpoints
│   │   └── middleware.go     # Auth, logging
│   ├── collector/
│   │   ├── filesystem.go     # FS traversal
│   │   ├── scanner.go        # File type detection
│   │   └── uploader.go       # Chunked upload
│   ├── serving/
│   │   ├── inference.go      # Model inference
│   │   ├── loader.go         # Model loading
│   │   └── cache.go          # Response caching
│   ├── config/
│   │   └── config.go         # JSON config handling
│   └── tunnel/
│       └── secure.go         # Secure backend comm
├── pkg/
│   ├── formats/              # File format handlers
│   └── crypto/               # Encryption utilities
└── configs/
    └── agent.json            # Default configuration
```

**Responsibilities:**
- Local filesystem scanning and data collection
- Secure upload to cloud backend
- Model downloading and local serving
- API endpoint for local inference
- Health reporting and telemetry

**Key Features:**
- Single binary, no dependencies
- Cross-compile for Windows/macOS/Linux
- Auto-update capability
- Encrypted communication (mTLS)
- Local model caching

### 2. Python ML Toolkit (Backend Services)

```
ads-ml/
├── ads_ml/
│   ├── __init__.py
│   ├── ingest/
│   │   ├── __init__.py
│   │   ├── documents.py      # PDF, DOCX, etc.
│   │   ├── images.py         # Image processing
│   │   ├── databases.py      # DB connectors
│   │   ├── ocr.py            # Text extraction
│   │   └── chunker.py        # Text chunking
│   ├── preprocess/
│   │   ├── __init__.py
│   │   ├── cleaning.py       # Data cleaning
│   │   ├── augmentation.py   # Data augmentation
│   │   ├── embedding.py      # Vector embeddings
│   │   └── validation.py     # Data validation
│   ├── training/
│   │   ├── __init__.py
│   │   ├── orchestrator.py   # Training coordinator
│   │   ├── fine_tuner.py     # Model fine-tuning
│   │   ├── evaluator.py      # Model evaluation
│   │   └── exporter.py       # Model export
│   ├── models/
│   │   ├── __init__.py
│   │   ├── classifier.py     # Classification models
│   │   ├── qa.py             # Q&A models
│   │   ├── generator.py      # Text generation
│   │   └── vision.py         # Vision models
│   ├── serving/
│   │   ├── __init__.py
│   │   ├── inference.py      # Inference engine
│   │   └── optimization.py   # Model optimization
│   └── utils/
│       ├── __init__.py
│       ├── config.py         # Configuration
│       └── metrics.py        # Logging/metrics
├── services/
│   ├── data_service/         # FastAPI service
│   ├── train_service/        # Training service
│   └── model_service/        # Model management
├── workers/
│   └── training_worker.py    # Celery worker
├── tests/
└── pyproject.toml
```

**Key Dependencies:**
- `transformers` - Hugging Face models
- `datasets` - Data handling
- `accelerate` - Distributed training
- `peft` - Parameter-efficient fine-tuning
- `bitsandbytes` - Quantization
- `fastapi` - API services
- `celery` - Task queue
- `pytesseract` - OCR
- `unstructured` - Document parsing

### 3. API Gateway

Technology: **Kong** or **Traefik** or custom Go gateway

```yaml
# API Routes
/api/v1/
  /auth/
    POST   /login
    POST   /register
    POST   /refresh
  /data/
    POST   /upload              # Initiate upload
    PUT    /upload/{id}/chunk   # Upload chunk
    POST   /upload/{id}/complete
    GET    /sources             # List data sources
    DELETE /sources/{id}
  /training/
    POST   /jobs                # Create training job
    GET    /jobs                # List jobs
    GET    /jobs/{id}           # Job status
    POST   /jobs/{id}/cancel    # Cancel job
  /models/
    GET    /                    # List models
    GET    /{id}                # Model details
    GET    /{id}/download       # Download model
    DELETE /{id}                # Delete model
  /inference/
    POST   /{model_id}/predict  # Run inference
  /billing/
    GET    /usage               # Current usage
    GET    /invoices            # Invoice history
    POST   /estimate            # Price estimate
```

### 4. Database Schema (PostgreSQL)

```sql
-- Users and Auth
CREATE TABLE users (
    id UUID PRIMARY KEY,
    email VARCHAR(255) UNIQUE NOT NULL,
    password_hash VARCHAR(255) NOT NULL,
    plan_id UUID REFERENCES plans(id),
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE plans (
    id UUID PRIMARY KEY,
    name VARCHAR(50) NOT NULL,
    tier VARCHAR(20) NOT NULL,  -- basic, indie, legal, medical, gov, finance, custom
    price_base DECIMAL(10,2),
    price_per_gb DECIMAL(10,4),
    price_per_gpu_hour DECIMAL(10,4),
    max_data_gb INTEGER,
    max_models INTEGER,
    features JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Data Sources
CREATE TABLE data_sources (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,  -- upload, filesystem, database
    status VARCHAR(20) NOT NULL, -- pending, processing, ready, error
    size_bytes BIGINT,
    file_count INTEGER,
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Training Jobs
CREATE TABLE training_jobs (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    data_source_ids UUID[],
    model_type VARCHAR(50) NOT NULL,
    config JSONB NOT NULL,
    status VARCHAR(20) NOT NULL, -- queued, training, completed, failed, cancelled
    progress INTEGER DEFAULT 0,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    error_message TEXT,
    cost_estimate DECIMAL(10,2),
    cost_actual DECIMAL(10,2),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Trained Models
CREATE TABLE models (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    job_id UUID REFERENCES training_jobs(id),
    name VARCHAR(255) NOT NULL,
    type VARCHAR(50) NOT NULL,
    version VARCHAR(20),
    size_bytes BIGINT,
    storage_path VARCHAR(500),  -- S3/OCI path
    hf_repo_id VARCHAR(255),    -- Hugging Face repo (private)
    metrics JSONB,              -- Accuracy, loss, etc.
    config JSONB,
    status VARCHAR(20) NOT NULL, -- ready, archived, deleted
    created_at TIMESTAMP DEFAULT NOW()
);

-- Usage Tracking
CREATE TABLE usage_events (
    id UUID PRIMARY KEY,
    user_id UUID REFERENCES users(id),
    event_type VARCHAR(50) NOT NULL,
    resource_id UUID,
    quantity DECIMAL(10,4),
    unit VARCHAR(20),
    cost DECIMAL(10,4),
    metadata JSONB,
    created_at TIMESTAMP DEFAULT NOW()
);
```

### 5. Message Queue (Redis + Celery)

```python
# Task definitions
@celery.task(bind=True)
def process_upload(self, upload_id: str):
    """Process uploaded data files"""
    pass

@celery.task(bind=True)
def train_model(self, job_id: str):
    """Execute model training"""
    pass

@celery.task(bind=True)
def export_model(self, model_id: str, format: str):
    """Export trained model"""
    pass

@celery.task
def cleanup_expired():
    """Cleanup expired data and models"""
    pass
```

### 6. Object Storage Structure

```
bucket: ads-model-maker/
├── uploads/
│   └── {user_id}/
│       └── {upload_id}/
│           ├── manifest.json
│           └── chunks/
├── processed/
│   └── {user_id}/
│       └── {source_id}/
│           ├── metadata.json
│           └── data/
├── models/
│   └── {user_id}/
│       └── {model_id}/
│           ├── config.json
│           ├── model.safetensors
│           ├── tokenizer/
│           └── metrics.json
└── exports/
    └── {user_id}/
        └── {export_id}/
```

## Communication Patterns

### Go Agent ↔ Backend

```
Agent                                    Backend
  │                                         │
  │──── mTLS Handshake ────────────────────▶│
  │◀─── Certificate Validation ─────────────│
  │                                         │
  │──── Register Agent ────────────────────▶│
  │◀─── Agent Token ────────────────────────│
  │                                         │
  │──── Upload Chunks (encrypted) ─────────▶│
  │◀─── ACK ────────────────────────────────│
  │                                         │
  │──── Poll for Model ────────────────────▶│
  │◀─── Model Download URL ─────────────────│
  │                                         │
  │──── Health Heartbeat ──────────────────▶│
  │◀─── Commands (optional) ────────────────│
```

### Training Pipeline Flow

```
1. User uploads data
   └─▶ Data Service receives chunks
       └─▶ Stores in Object Storage
           └─▶ Triggers processing task

2. Processing task
   └─▶ Validates data format
       └─▶ Runs preprocessing
           └─▶ Stores processed data
               └─▶ Updates data_source status

3. User creates training job
   └─▶ Train Service validates config
       └─▶ Calculates cost estimate
           └─▶ Creates job record
               └─▶ Enqueues training task

4. Training task
   └─▶ Worker picks up job
       └─▶ Loads processed data
           └─▶ Initializes base model (HF)
               └─▶ Runs fine-tuning
                   └─▶ Evaluates model
                       └─▶ Exports to storage
                           └─▶ Updates job status

5. Model ready
   └─▶ User downloads or deploys
       └─▶ Agent fetches model
           └─▶ Serves locally via API
```

## Deployment Architecture

### Development
- Docker Compose for local development
- Single-node setup

### Staging
- Kubernetes (k3s or managed)
- Shared GPU resources
- Limited storage

### Production
```
┌─────────────────────────────────────────────────────────────┐
│                    Load Balancer (L7)                       │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   API Gateway   │  │   API Gateway   │  │   API Gateway   │
│    (Node 1)     │  │    (Node 2)     │  │    (Node 3)     │
└─────────────────┘  └─────────────────┘  └─────────────────┘
         │                    │                    │
         └────────────────────┼────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Data Service   │  │  Train Service  │  │  Model Service  │
│   (Replicas)    │  │   (Replicas)    │  │   (Replicas)    │
└─────────────────┘  └─────────────────┘  └─────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                    GPU Training Cluster                      │
│  ┌───────────┐  ┌───────────┐  ┌───────────┐  ┌───────────┐│
│  │  Worker   │  │  Worker   │  │  Worker   │  │  Worker   ││
│  │  (GPU)    │  │  (GPU)    │  │  (GPU)    │  │  (GPU)    ││
│  └───────────┘  └───────────┘  └───────────┘  └───────────┘│
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│   PostgreSQL    │  │     Redis       │  │  Object Storage │
│   (Primary +    │  │   (Cluster)     │  │    (S3/OCI)     │
│    Replica)     │  │                 │  │                 │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

## Security Considerations

1. **Data Encryption**
   - At rest: AES-256
   - In transit: TLS 1.3
   - End-to-end for sensitive tiers (HIPAA, Gov)

2. **Authentication**
   - JWT with short expiry
   - Refresh token rotation
   - MFA for enterprise tiers

3. **Authorization**
   - RBAC for team accounts
   - Resource-level permissions
   - API key scoping

4. **Audit Logging**
   - All data access logged
   - Training job audit trail
   - Compliance reporting

5. **Network Isolation**
   - VPC for production
   - Private subnets for services
   - WAF for public endpoints
