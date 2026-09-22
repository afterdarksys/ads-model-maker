# Serving Layer & API Design

## Overview

The serving layer provides multiple deployment options:
1. **Local Serving** - Go agent serves models on user's machine
2. **Cloud Serving** - Hosted inference via aiserve.farm
3. **Hybrid** - Local with cloud fallback

## Deployment Options

### Option 1: Local Serving (Go Agent)

```
User App ──► Go Agent (localhost:8484) ──► Local Model
                │
                └── MCP Protocol (for LLM tools)
```

**Pros:**
- No latency to cloud
- Data stays local (privacy/compliance)
- No per-request costs
- Works offline

**Cons:**
- Requires local compute resources
- Model size limited by local hardware
- User manages agent lifecycle

### Option 2: Cloud Serving (aiserve.farm)

```
User App ──► API Gateway ──► Load Balancer ──► Inference Cluster
                                                    │
                                                    └── GPU Nodes
```

**Pros:**
- Zero local requirements
- Unlimited scale
- Always-on availability
- Large model support

**Cons:**
- Per-request costs
- Network latency
- Data leaves user's control

### Option 3: Hybrid

```
User App ──► Go Agent ──► Local Model (if available)
                │
                └── Cloud Fallback (if local unavailable)
```

**Pros:**
- Best of both worlds
- Graceful degradation
- Cost optimization

## API Specification

### Base URL
- Local: `http://localhost:8484/api/v1`
- Cloud: `https://api.aiserve.farm/v1`

### Authentication

```http
# API Key (header)
Authorization: Bearer ads_sk_xxxxxxxxxxxxx

# Or query parameter (for webhooks)
?api_key=ads_sk_xxxxxxxxxxxxx
```

### Rate Limits

| Plan | Requests/min | Requests/day | Concurrent |
|------|-------------|--------------|------------|
| Basic | 60 | 1,000 | 5 |
| Indie | 300 | 10,000 | 10 |
| Professional | 1,000 | 100,000 | 25 |
| Enterprise | Unlimited | Unlimited | 100 |

### Endpoints

#### Health & Status

```http
GET /health
```

Response:
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "uptime": "24h12m",
  "models_loaded": 3
}
```

#### Inference Endpoints

##### Predict (Generic)

```http
POST /inference/predict
Content-Type: application/json

{
  "model_id": "model_abc123",
  "input": {
    "text": "This product is amazing!"
  },
  "options": {
    "return_probabilities": true
  }
}
```

Response:
```json
{
  "id": "pred_xyz789",
  "model_id": "model_abc123",
  "result": {
    "label": "positive",
    "confidence": 0.94,
    "probabilities": {
      "positive": 0.94,
      "negative": 0.04,
      "neutral": 0.02
    }
  },
  "latency_ms": 12,
  "cached": false
}
```

##### Classification

```http
POST /inference/classify
Content-Type: application/json

{
  "model_id": "model_abc123",
  "texts": [
    "Great product, highly recommend!",
    "Terrible experience, never again.",
    "It's okay, nothing special."
  ],
  "options": {
    "batch_size": 10,
    "return_embeddings": false
  }
}
```

Response:
```json
{
  "id": "cls_xyz789",
  "results": [
    {"text": "Great product...", "label": "positive", "confidence": 0.96},
    {"text": "Terrible experience...", "label": "negative", "confidence": 0.92},
    {"text": "It's okay...", "label": "neutral", "confidence": 0.78}
  ],
  "latency_ms": 45,
  "tokens_processed": 42
}
```

##### Question Answering

```http
POST /inference/qa
Content-Type: application/json

{
  "model_id": "model_qa123",
  "question": "What is the return policy?",
  "context": "Our return policy allows returns within 30 days of purchase...",
  "options": {
    "top_k": 3,
    "min_confidence": 0.5
  }
}
```

Response:
```json
{
  "id": "qa_xyz789",
  "answers": [
    {
      "text": "30 days of purchase",
      "confidence": 0.89,
      "start": 42,
      "end": 61
    }
  ],
  "latency_ms": 28
}
```

##### Embeddings

```http
POST /inference/embed
Content-Type: application/json

{
  "model_id": "model_embed123",
  "texts": [
    "How do I reset my password?",
    "Password reset instructions"
  ],
  "options": {
    "normalize": true,
    "truncate": true
  }
}
```

Response:
```json
{
  "id": "emb_xyz789",
  "embeddings": [
    [0.021, -0.045, 0.089, ...],
    [0.019, -0.043, 0.092, ...]
  ],
  "dimensions": 384,
  "latency_ms": 15
}
```

##### Text Generation

```http
POST /inference/generate
Content-Type: application/json

{
  "model_id": "model_gen123",
  "prompt": "Write a professional email response to:",
  "input": "Customer complaint about delayed shipment",
  "options": {
    "max_tokens": 200,
    "temperature": 0.7,
    "stop_sequences": ["\n\n"]
  }
}
```

Response:
```json
{
  "id": "gen_xyz789",
  "text": "Dear Valued Customer,\n\nThank you for reaching out...",
  "tokens_generated": 145,
  "finish_reason": "stop",
  "latency_ms": 450
}
```

##### Semantic Search

```http
POST /inference/search
Content-Type: application/json

{
  "model_id": "model_search123",
  "query": "how to change password",
  "options": {
    "top_k": 5,
    "min_score": 0.7,
    "filter": {
      "category": "account"
    }
  }
}
```

Response:
```json
{
  "id": "search_xyz789",
  "results": [
    {
      "id": "doc_123",
      "text": "To change your password, go to Settings...",
      "score": 0.94,
      "metadata": {"category": "account", "source": "faq.md"}
    },
    {
      "id": "doc_456",
      "text": "Password requirements include...",
      "score": 0.82,
      "metadata": {"category": "account", "source": "security.md"}
    }
  ],
  "latency_ms": 23
}
```

#### Model Management

##### List Models

```http
GET /models
```

Response:
```json
{
  "models": [
    {
      "id": "model_abc123",
      "name": "Support Ticket Classifier",
      "type": "classification",
      "status": "ready",
      "size_bytes": 125000000,
      "created_at": "2024-01-15T10:00:00Z",
      "metrics": {
        "accuracy": 0.94,
        "f1_score": 0.92
      }
    }
  ],
  "total": 3
}
```

##### Get Model Details

```http
GET /models/{model_id}
```

Response:
```json
{
  "id": "model_abc123",
  "name": "Support Ticket Classifier",
  "type": "classification",
  "status": "ready",
  "config": {
    "base_model": "distilbert-base-uncased",
    "num_labels": 5,
    "labels": ["billing", "technical", "shipping", "account", "other"]
  },
  "metrics": {
    "accuracy": 0.94,
    "f1_score": 0.92,
    "precision": 0.93,
    "recall": 0.91
  },
  "usage": {
    "total_predictions": 15420,
    "last_used": "2024-01-20T15:30:00Z"
  }
}
```

##### Download Model

```http
GET /models/{model_id}/download?format=safetensors
```

Response: Binary file stream or signed URL

##### Load/Unload Model (Local Agent)

```http
POST /models/{model_id}/load
POST /models/{model_id}/unload
```

#### Streaming Endpoints

For generation models, support Server-Sent Events:

```http
POST /inference/generate/stream
Content-Type: application/json
Accept: text/event-stream

{
  "model_id": "model_gen123",
  "prompt": "Write a story about..."
}
```

Response (SSE):
```
event: token
data: {"token": "Once", "index": 0}

event: token
data: {"token": " upon", "index": 1}

event: token
data: {"token": " a", "index": 2}

event: done
data: {"tokens_generated": 150, "finish_reason": "stop"}
```

## MCP Protocol Integration

The Go agent implements MCP for LLM tool use:

### Tool Discovery

```http
GET /mcp/v1/tools
```

Response:
```json
{
  "tools": [
    {
      "name": "classify_text",
      "description": "Classify text into predefined categories",
      "inputSchema": {
        "type": "object",
        "properties": {
          "text": {"type": "string"},
          "model_id": {"type": "string"}
        },
        "required": ["text"]
      }
    },
    {
      "name": "semantic_search",
      "description": "Search documents using semantic similarity",
      "inputSchema": {
        "type": "object",
        "properties": {
          "query": {"type": "string"},
          "top_k": {"type": "integer", "default": 5}
        },
        "required": ["query"]
      }
    }
  ]
}
```

### Tool Invocation

```http
POST /mcp/v1/tools/classify_text/call
Content-Type: application/json

{
  "arguments": {
    "text": "My order hasn't arrived yet",
    "model_id": "model_abc123"
  }
}
```

Response:
```json
{
  "content": [
    {
      "type": "text",
      "text": "Classification: shipping (confidence: 0.91)"
    }
  ]
}
```

### WebSocket MCP

```javascript
// Client example
const ws = new WebSocket('ws://localhost:8484/mcp');

ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 1,
  method: 'tools/list'
}));

ws.send(JSON.stringify({
  jsonrpc: '2.0',
  id: 2,
  method: 'tools/call',
  params: {
    name: 'classify_text',
    arguments: {
      text: 'My order is late'
    }
  }
}));
```

## SDK Examples

### Python SDK

```python
from ads_model_maker import Client

# Initialize client
client = Client(
    api_key="ads_sk_xxx",
    base_url="http://localhost:8484"  # or https://api.aiserve.farm
)

# Classification
result = client.classify(
    model_id="model_abc123",
    texts=["Great product!", "Terrible service"]
)
print(result.labels)  # ['positive', 'negative']

# Question answering
answer = client.qa(
    model_id="model_qa123",
    question="What's the return policy?",
    context=document_text
)
print(answer.text)

# Embeddings
embeddings = client.embed(
    model_id="model_embed123",
    texts=["query 1", "query 2"]
)
# Returns numpy array

# Semantic search
results = client.search(
    model_id="model_search123",
    query="password reset",
    top_k=5
)
for r in results:
    print(f"{r.score}: {r.text}")
```

### JavaScript/TypeScript SDK

```typescript
import { ADSClient } from '@ads-model-maker/sdk';

const client = new ADSClient({
  apiKey: 'ads_sk_xxx',
  baseUrl: 'http://localhost:8484'
});

// Classification
const result = await client.classify({
  modelId: 'model_abc123',
  texts: ['Great product!', 'Terrible service']
});
console.log(result.labels);

// Streaming generation
const stream = client.generateStream({
  modelId: 'model_gen123',
  prompt: 'Write a story...'
});

for await (const chunk of stream) {
  process.stdout.write(chunk.token);
}
```

### Go SDK

```go
package main

import (
    ads "github.com/ads-model-maker/go-sdk"
)

func main() {
    client := ads.NewClient(
        ads.WithAPIKey("ads_sk_xxx"),
        ads.WithBaseURL("http://localhost:8484"),
    )

    // Classification
    result, err := client.Classify(ctx, &ads.ClassifyRequest{
        ModelID: "model_abc123",
        Texts:   []string{"Great product!", "Terrible service"},
    })

    // Question answering
    answer, err := client.QA(ctx, &ads.QARequest{
        ModelID:  "model_qa123",
        Question: "What's the return policy?",
        Context:  documentText,
    })
}
```

### cURL Examples

```bash
# Health check
curl http://localhost:8484/health

# Classification
curl -X POST http://localhost:8484/api/v1/inference/classify \
  -H "Authorization: Bearer ads_sk_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_abc123",
    "texts": ["Great product!", "Terrible service"]
  }'

# Question answering
curl -X POST http://localhost:8484/api/v1/inference/qa \
  -H "Authorization: Bearer ads_sk_xxx" \
  -H "Content-Type: application/json" \
  -d '{
    "model_id": "model_qa123",
    "question": "What is the return policy?",
    "context": "Our return policy allows returns within 30 days..."
  }'
```

## Error Handling

### Error Response Format

```json
{
  "error": {
    "code": "model_not_found",
    "message": "Model 'model_xyz' not found or not loaded",
    "details": {
      "model_id": "model_xyz",
      "available_models": ["model_abc123", "model_def456"]
    }
  },
  "request_id": "req_abc123"
}
```

### Error Codes

| Code | HTTP Status | Description |
|------|-------------|-------------|
| `invalid_request` | 400 | Malformed request body |
| `authentication_failed` | 401 | Invalid or missing API key |
| `permission_denied` | 403 | Not authorized for this resource |
| `model_not_found` | 404 | Model doesn't exist |
| `model_not_loaded` | 503 | Model exists but not loaded |
| `rate_limit_exceeded` | 429 | Too many requests |
| `inference_failed` | 500 | Model inference error |
| `timeout` | 504 | Request timed out |

## Testing Tools

### Built-in Playground

The Go agent includes a web-based testing interface:

```
http://localhost:8484/playground
```

Features:
- Interactive model testing
- Request/response inspection
- Latency measurement
- Batch testing
- Export curl commands

### API Testing Script

```python
# test_api.py
import requests
import time

BASE_URL = "http://localhost:8484/api/v1"
API_KEY = "ads_sk_xxx"

def test_classification():
    """Test classification endpoint"""
    response = requests.post(
        f"{BASE_URL}/inference/classify",
        headers={"Authorization": f"Bearer {API_KEY}"},
        json={
            "model_id": "model_abc123",
            "texts": ["Test input"]
        }
    )
    assert response.status_code == 200
    result = response.json()
    assert "results" in result
    print(f"✓ Classification: {result['latency_ms']}ms")

def test_latency(n=100):
    """Measure average latency"""
    times = []
    for _ in range(n):
        start = time.time()
        requests.post(
            f"{BASE_URL}/inference/classify",
            headers={"Authorization": f"Bearer {API_KEY}"},
            json={"model_id": "model_abc123", "texts": ["Test"]}
        )
        times.append(time.time() - start)

    avg = sum(times) / len(times) * 1000
    p95 = sorted(times)[int(n * 0.95)] * 1000
    print(f"Average: {avg:.2f}ms, P95: {p95:.2f}ms")

if __name__ == "__main__":
    test_classification()
    test_latency()
```

## Monitoring & Observability

### Prometheus Metrics

```
# Endpoint: GET /metrics

# Request metrics
ads_requests_total{endpoint="/inference/classify",status="200"} 15420
ads_request_duration_seconds{endpoint="/inference/classify",quantile="0.95"} 0.025

# Model metrics
ads_model_loaded{model_id="model_abc123"} 1
ads_model_inference_total{model_id="model_abc123"} 15420
ads_model_inference_duration_seconds{model_id="model_abc123",quantile="0.95"} 0.012

# Cache metrics
ads_cache_hits_total 12500
ads_cache_misses_total 2920
ads_cache_size_bytes 524288000

# System metrics
ads_memory_usage_bytes 1073741824
ads_goroutines 45
```

### Structured Logging

```json
{
  "level": "info",
  "time": "2024-01-20T15:30:00Z",
  "request_id": "req_abc123",
  "endpoint": "/inference/classify",
  "model_id": "model_abc123",
  "latency_ms": 12,
  "status": 200,
  "tokens": 25
}
```
