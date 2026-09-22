# ADS Model Maker

Toolkit for turning a pile of documents into a small model you can serve locally. The Python package ingests and trains. The Go agent loads a finished model and answers HTTP and MCP requests.

This repository is the library and the agent. The hosted product described in [plans/](plans/) (accounts, billing, a web UI) is not built yet.

## Layout

| Path | What it is |
| --- | --- |
| `ads-ml/` | Ingest, chunk, train, and the security-text classifier |
| `ads-agent/` | Local model server |
| `plans/` | Design notes for the larger platform |
| `docker-compose.yml` | Local Postgres, Redis, and MinIO for later services |

## Python toolkit

Requires Python 3.11 or newer.

```bash
cd ads-ml
pip install -e ".[dev]"
PYTHONPATH=. python -m pytest tests/test_review_fixes.py
```

What you can do today:

- Extract text from TXT, PDF, DOCX, HTML, CSV, and JSON, then chunk it.
- Keep a local corpus and add, update, remove, or query records.
- Train a classifier, embedder, or generator from a JSON job file.
- Report parameter count, disk size, and an activation RAM estimate before a run.
- Build a portable security classifier for phishing email, alert severity, log category, or secret exposure.

A security training file is JSONL. Each line has `text` and `label`.

```python
from ads_ml.security import build_security_model

build_security_model("alerts.jsonl", "phishing_email", "output/phish")
```

That writes `model.ads.json`, `MODEL_CARD.md`, and `report.json`. Secrets that match the built-in patterns are removed before training. The score is for an analyst queue, not an allow-or-block decision.

## Go agent

```bash
cd ads-agent
go test ./internal/inference/ ./internal/api/ ./internal/config/
go run ./cmd/agent
```

The server listens on `127.0.0.1:8484`. Set `ADS_SERVER_HOST` and `ADS_SERVER_PORT` to change that. A non-loopback address is refused unless `ADS_SERVER_API_KEY` is set. When the key is set, inference, model load, metrics, and MCP require `Authorization: Bearer <key>` or `X-API-Key`. `/health` and `/ready` stay open.

The agent scores `*.ads.json` models produced by the security builder. Put the file in the model directory (`~/.ads-agent/models` by default) and load it with the file name as the id:

```bash
curl -s -X POST localhost:8484/api/v1/models/model.ads.json/load \
  -H 'Content-Type: application/json' \
  -d '{"path":"'"$HOME"'/.ads-agent/models/model.ads.json"}'
```

ONNX and GoLearn files can be loaded, but those runtimes still return a placeholder label.

## Design notes

Start with [plans/00-overview.md](plans/00-overview.md). [plans/09-review-and-roadmap.md](plans/09-review-and-roadmap.md) records the current bugs that were fixed and the features still open.
