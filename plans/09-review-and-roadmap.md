# Review, fixes, and the next feature set

Reviewed on 2026-09-22 against the Python training toolkit and the Go serving agent.

## Bugs that were breaking real use

Training and ingest:

- A paragraph longer than the chunk window was stored as one chunk. Token chunking could fail to make progress when overlap was at least the window size, because `text[-0:]` is the whole string, so an overlap of zero copied the previous chunk forward.
- The last short sentence of a document was dropped once an earlier chunk existed.
- `auto_chunk(..., strategy="fixed")` silently used the semantic chunker. An unknown strategy now fails.
- Character padding used id 0 and an all-ones attention mask, and generation labels were trained to predict that padding.
- An unknown class name crashed with a raw `KeyError` inside the dataset.
- `train_classifier` could build an empty training split on a tiny dataset, and the cosine schedule rejected a zero step count.
- Embedder presets passed `heads` into a config field named `num_heads`, so `create_embedder` could not build a preset.
- Contrastive loss treated a token row whose length happened to equal the embedding size as an embedding and skipped the encoder.
- Batched generation called `.item()` on a whole batch and crashed. Streaming now requires one row.
- ONNX export asked for a logits tensor while `forward` returned a dict.
- Content hashes were truncated to 16 hex characters.
- Local storage treated `/data` as a parent of `/data-evil`.
- `ads_ml.ingest` imported a pipeline module that did not exist.
- Vast.ai put the API key on the process command line, and file copy disabled SSH host-key checking.

The agent:

- Any file extension was loaded as ONNX and then served a placeholder label.
- Prediction looked up the model map without holding the lock, and an empty model id picked an arbitrary loaded model.
- Replacing a model did not close the previous runtime.
- Cache hit counters raced, and `cache_entries` reported lookups rather than entries stored.
- The rate limiter keyed off a caller-supplied `X-API-Key`, so a client could skip the limit by rotating that header.
- Model load accepted any filesystem path and echoed the internal error.
- The value-log garbage collector kept running after the database closed.
- WebSocket `tools/list` returned no tools while the HTTP list did.
- `ADS_SERVER_HOST` and `ADS_SERVER_PORT` were not applied, because Viper does not fill nested keys from the environment on unmarshal.
- The pinned JSON library did not link on the current Go toolchain. Request and storage JSON now uses the standard library.
- A dependency commit for `savsgio/gotils` is gone from the module proxy. `go.mod` replaces it with a commit that still resolves.

Listening on a non-loopback address without `ADS_SERVER_API_KEY` is refused. With a key set, inference, model load, metrics, and MCP require `Authorization: Bearer` or `X-API-Key`. `/health` and `/ready` stay open. The comparison hashes both values and compares them in constant time.

Threats: this stops anonymous use of a deployed agent and stops a load path from escaping the model directory. It does not stop a caller who already has the key, and it does not encrypt the body. SSH copy now requires a known host key, so the first connection to a new Vast.ai host fails until that key is trusted.

## Design problems that are still open

The character encoder is not a tokenizer. A transformer trained here will not transfer to normal model tooling.

The ONNX and GoLearn runtimes still return placeholder labels. The path that actually scores text is the `ads-linear-v1` file produced by the security builder.

There is no web UI. `web/src` is empty, and the agent does not remember loaded models across restarts.

`get_instance` in the Vast.ai provider still turns every failure into "not found".

The password labeler docstring talks about entropy thresholds. The number it compares is a composite score.

## First enhancement round, now in the toolkit

These are the gaps the README and the MVP roadmap already called for, and they do not require a GPU or a hosted dataset.

- A local corpus with add, update, remove, and query, plus a content hash so the same text is stored once (`ads_ml/corpus.py`).
- A footprint report: parameter count, bytes at rest, and an activation RAM estimate (`ads_ml/report.py`).
- A JSON job file that rejects unknown keys and a classifier with fewer than two labels (`ads_ml/training/job_config.py`).
- CSV and JSON ingestion, and an ingest pipeline that extracts a file and chunks it.
- Classifier evaluation reports macro F1 alongside accuracy.

## A. Ten features for someone making models in general

1. Corpus add, update, remove, and query. Done in this round.
2. A footprint report before a run starts: parameters, disk, RAM estimate, and how much text is in the set. Done in this round.
3. A JSON job file shared by a future UI and the CLI. Done in this round.
4. CSV and JSON alongside PDF, DOCX, HTML, and text. Done in this round.
5. Holdout splits that refuse to put near-duplicate text on both sides. The security builder splits by class. The general trainer still uses a shuffled 90/10 cut.
6. A data recipe: hash the corpus plus the job file and store that hash next to the checkpoint so a run can be repeated.
7. Resume from the last checkpoint, including optimizer state. Checkpoints are written today and are not reloaded.
8. A real tokenizer, trained on the corpus, instead of raw code points.
9. Side-by-side eval of two checkpoints on the same held-out file, with the delta called out per label.
10. A cost preview: estimated step count, activation RAM, and whether the chosen preset fits the machine before training starts.

## B. Ten features for someone whose models are about IT security

The product for this user is `ads_ml.security`. It trains a portable text classifier for one of four tasks: phishing email, alert severity, log category, or secret exposure. The agent loads the resulting `model.ads.json` and scores it. It does not detonate files, scan networks, or make an allow-or-block decision.

1. Task presets and a JSONL loader that rejects labels outside the task. Done.
2. Secret redaction before text becomes features, so keys and private-key blocks are not learned. Done. It will miss secrets that do not match the patterns.
3. Per-class precision, recall, and F1, because accuracy hides a model that never predicts the rare class. Done.
4. A threshold sweep for the positive class, so a queue can pick a cutoff instead of taking argmax. Done.
5. A portable linear model the agent can score without ONNX. Done.
6. A model card with intended use, out of scope, data counts, metrics, and footprint. Done.
7. Alert-to-label review with disagreements stored when two analysts mark the same text differently. Not built.
8. Time-based splits so a model is tested on alerts newer than the ones it trained on. Not built.
9. A feedback join from the agent: store the text, the predicted label, and the analyst's correction, and export that as the next JSONL. Not built.
10. Separate operating points per severity, with a stated false-positive budget for paging a human. The sweep reports the curve. It does not yet enforce a budget.

## C. Five enhancements added on top of that security product

1. Near-duplicate suppression. Repeated tickets are dropped with a 64-bit simhash before training so one pasted alert cannot dominate a class.
2. Defanged indicator normalization. `hxxp` and `[.]` are rewritten to ordinary URL spelling so the model sees the same shape analysts see after they refang a report. This does not resolve DNS or fetch the URL.
3. A calibration table. Predictions are grouped by confidence and compared with how often that bucket was right.
4. An active-learning queue. The eval examples with the smallest gap between the top two classes are written into `report.json` for a person to label next.
5. A regression gate. A new run fails closed when macro F1 is missing, and it fails when macro F1 drops further than the allowed tolerance from a baseline report.

`build_security_model` writes `model.ads.json`, `MODEL_CARD.md`, and `report.json` into the output directory. The card states that the score is not a block decision.

## What was verified

`PYTHONPATH=ads-ml python3.11 -m pytest ads-ml/tests/test_review_fixes.py` — 19 passed.

`go test` in `ads-agent` for `internal/inference`, `internal/api`, and `internal/config`. The API tests cover a wrong key, a missing key, an open health check, a non-loopback bind without a key, a model path outside the model directory, and an oversized body.
