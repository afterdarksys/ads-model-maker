"""Regression tests for the review fixes and the security model tools."""

import json
from pathlib import Path

import pytest

from ads_ml.corpus import Corpus
from ads_ml.ingest.chunker import SentenceChunker, TokenChunker, auto_chunk
from ads_ml.ingest.documents import extract_document
from ads_ml.ingest.pipeline import IngestPipeline
from ads_ml.report import model_footprint
from ads_ml.security.active import review_queue
from ads_ml.security.build import build_security_model
from ads_ml.security.gate import regression_gate
from ads_ml.security.ioc import normalize_defanged
from ads_ml.security.linear import LinearTextClassifier
from ads_ml.security.metrics import calibration_buckets, classification_report, threshold_sweep
from ads_ml.security.prepare import prepare_security_jsonl
from ads_ml.security.redact import redact_text
from ads_ml.storage.local_storage import LocalStorage
from ads_ml.training.dataset import ClassificationDataset, GenerationDataset
from ads_ml.training.job_config import load_job_config


class _ListTokenizer:
    def encode(self, text):
        return list(range(len(text)))

    def decode(self, tokens):
        return "x" * len(tokens)


def test_semantic_chunker_splits_a_long_paragraph():
    text = "word " * 800
    chunks = auto_chunk(text, strategy="semantic", max_chars=120, min_chars=20, overlap_chars=0)
    assert len(chunks) > 1
    assert all(len(chunk.text) <= 120 for chunk in chunks)
    assert sum(len(chunk.text) for chunk in chunks) > 120


def test_token_chunker_finishes_when_overlap_exceeds_window():
    chunks = TokenChunker(_ListTokenizer(), max_tokens=4, overlap_tokens=10).chunk("abcdefghijklmnop")
    assert chunks
    assert len(chunks) < 20


def test_sentence_chunker_keeps_a_short_tail():
    text = ("A" * 40) + ". " + ("B" * 40) + ". Tail"
    chunks = SentenceChunker(max_chars=45, min_chars=30, overlap_sentences=0).chunk(text)
    assert "Tail" in " ".join(chunk.text for chunk in chunks)


def test_fixed_strategy_and_unknown_strategy():
    chunks = auto_chunk("abcdef", strategy="fixed", max_chars=2, overlap_chars=0)
    assert [chunk.text for chunk in chunks] == ["ab", "cd", "ef"]
    with pytest.raises(ValueError):
        auto_chunk("abcdef", strategy="mystery")


def test_generation_labels_ignore_padding():
    item = GenerationDataset(["ab"], max_length=5)[0]
    assert item["attention_mask"].tolist() == [1, 1, 0, 0, 0]
    assert item["labels"][:2].tolist() != [-100, -100]
    assert item["labels"][2:].tolist() == [-100, -100, -100]


def test_unknown_classification_label_is_rejected():
    dataset = ClassificationDataset(["hello"], ["nope"], ["yes", "no"])
    with pytest.raises(KeyError):
        _ = dataset[0]


def test_local_storage_rejects_a_sibling_prefix():
    import tempfile

    with tempfile.TemporaryDirectory() as raw:
        root = Path(raw)
        storage = LocalStorage(root / "data")
        (root / "data2").mkdir()
        with pytest.raises(ValueError):
            storage._resolve_path("../data2/secret")


def test_corpus_add_update_remove_query(tmp_path):
    corpus = Corpus(tmp_path / "corpus.jsonl")
    added = corpus.add("Alpha alert", metadata={"source": "queue"})
    again = corpus.add("Alpha alert")
    assert again.id == added.id
    corpus.update(added.id, metadata={"source": "reviewed"})
    assert corpus.query(metadata={"source": "reviewed"})[0].id == added.id
    assert corpus.remove(added.id)
    assert corpus.stats()["documents"] == 0


def test_footprint_rejects_negative_parameters():
    report = model_footprint(num_params=100, dtype_bytes=4, hidden_dim=8, depth=2, seq_len=16)
    assert report["disk_bytes"] == 400
    assert report["estimated_peak_ram_bytes"] > report["disk_bytes"]
    with pytest.raises(ValueError):
        model_footprint(num_params=-1)


def test_job_config_rejects_unknown_keys_and_missing_labels(tmp_path):
    path = tmp_path / "job.json"
    path.write_text(json.dumps({
        "model_type": "classifier",
        "labels": ["benign", "phishing"],
        "data_path": "data.jsonl",
        "extra": True,
    }))
    with pytest.raises(ValueError):
        load_job_config(path)
    path.write_text(json.dumps({"model_type": "classifier", "labels": ["only"], "data_path": "data.jsonl"}))
    with pytest.raises(ValueError):
        load_job_config(path)


def test_csv_and_json_ingest_and_pipeline(tmp_path):
    csv_path = tmp_path / "rows.csv"
    csv_path.write_text("name,note\nalice,hello\n")
    csv_doc = extract_document(csv_path)
    assert "alice" in csv_doc.full_text
    assert len(csv_doc.file_hash) == 64

    json_path = tmp_path / "rows.json"
    json_path.write_text(json.dumps([{"text": "first record"}, {"text": "second record"}]))
    json_doc = extract_document(json_path)
    assert json_doc.pages[0].text == "first record"
    assert json_doc.pages[1].text == "second record"

    text_path = tmp_path / "note.txt"
    text_path.write_text("A short note for the pipeline.")
    result = IngestPipeline(strategy="fixed", max_chars=12, overlap_chars=0).ingest_file(text_path)
    assert result.chunks
    assert result.text.startswith("A short note")


def test_secret_redaction_does_not_keep_the_secret():
    source = "key=AKIAIOSFODNN7EXAMPLE and bearer sk-live-secret-value"
    result = redact_text(source)
    assert "AKIAIOSFODNN7EXAMPLE" not in result.text
    assert "sk-live-secret-value" not in result.text
    assert result.removed >= 2


def test_defang_normalization_and_duplicate_drop(tmp_path):
    assert normalize_defanged("hxxps://evil[.]example") == "https://evil.example"
    path = tmp_path / "alerts.jsonl"
    path.write_text(
        "\n".join([
            json.dumps({"text": "reset your password at hxxp://phish[.]example", "label": "phishing"}),
            json.dumps({"text": "reset your password at hxxp://phish[.]example", "label": "phishing"}),
            json.dumps({"text": "quarterly budget meeting notes for finance", "label": "benign"}),
            json.dumps({"text": "token=AKIAIOSFODNN7EXAMPLE", "label": "phishing"}),
        ]) + "\n"
    )
    prepared = prepare_security_jsonl(path, "phishing_email")
    assert prepared.duplicates_removed == 1
    assert prepared.redaction_counts.get("aws_access_key", 0) == 1
    assert all("AKIA" not in record["text"] for record in prepared.records)
    assert any("http://phish.example" in record["text"] for record in prepared.records)


def test_metrics_threshold_calibration_queue_and_gate():
    report = classification_report(
        ["phishing", "benign", "phishing"],
        ["phishing", "phishing", "phishing"],
        ["benign", "phishing"],
    )
    assert report["per_class"][1]["recall"] == 1
    sweep = threshold_sweep([True, False, True], [0.9, 0.2, 0.6])
    assert sweep[0]["threshold"] == 0.1
    buckets = calibration_buckets([0.95, 0.1], [True, False], bins=2)
    assert buckets[1]["count"] == 1
    queue = review_queue([
        {"text": "sure", "probabilities": {"a": 0.9, "b": 0.1}},
        {"text": "unsure", "probabilities": {"a": 0.51, "b": 0.49}},
    ])
    assert queue[0]["text"] == "unsure"
    failed = regression_gate({"macro_f1": 0.4}, {"macro_f1": 0.9}, max_drop=0.02)
    assert failed["pass"] is False
    assert regression_gate({"accuracy": 1}, {"macro_f1": 1})["pass"] is False


def test_linear_classifier_separates_two_phrases(tmp_path):
    classifier = LinearTextClassifier(["benign", "phishing"], dim=64, epochs=8, learning_rate=0.4, l2=0.0)
    texts = ["click here to reset your password"] * 6 + ["quarterly budget meeting notes"] * 6
    labels = ["phishing"] * 6 + ["benign"] * 6
    classifier.fit(texts, labels)
    assert classifier.predict("click here to reset your password")["label"] == "phishing"
    assert classifier.predict("quarterly budget meeting notes")["label"] == "benign"
    path = tmp_path / "model.ads.json"
    classifier.save(path)
    loaded = LinearTextClassifier.load(path)
    assert loaded.predict("quarterly budget meeting notes")["label"] == "benign"


def test_security_build_writes_card_queue_and_gate(tmp_path):
    path = tmp_path / "train.jsonl"
    rows = []
    for _ in range(4):
        rows.append({"text": "verify your account at hxxp://login[.]example now", "label": "phishing"})
        rows.append({"text": "the board approved the quarterly facilities budget", "label": "benign"})
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n")
    report = build_security_model(
        path,
        "phishing_email",
        tmp_path / "out",
        baseline_metrics={"macro_f1": 0.0},
        dim=64,
        epochs=6,
    )
    assert report["gate"]["pass"] is True
    assert (tmp_path / "out" / "model.ads.json").is_file()
    card = (tmp_path / "out" / "MODEL_CARD.md").read_text()
    assert "phishing_email" in card
    assert "Out of scope" in card
    assert report["review_queue"]


def test_trainer_rejects_an_empty_dataset():
    torch = pytest.importorskip("torch")
    from ads_ml.models.classifier import create_classifier
    from ads_ml.training.dataset import ClassificationDataset
    from ads_ml.training.trainer import ADSTrainer, TrainingConfig

    model = create_classifier(2, ["a", "b"], size="micro")
    empty = ClassificationDataset([], [], ["a", "b"])
    trainer = ADSTrainer(model, TrainingConfig(epochs=1, batch_size=2), empty)
    with pytest.raises(ValueError):
        trainer.train()


def test_create_embedder_accepts_a_preset():
    pytest.importorskip("torch")
    from ads_ml.models.embedder import create_embedder

    model = create_embedder(size="micro", embed_dim=32)
    assert model.config.num_heads > 0


def test_batched_generation_does_not_crash():
    torch = pytest.importorskip("torch")
    from ads_ml.models.generator import create_generator

    model = create_generator(size="micro")
    model.eval()
    tokens = torch.randint(1, 50, (2, 4))
    generated = model.generate(tokens, max_new_tokens=2, stop_tokens=[1])
    assert generated.shape[0] == 2
    assert generated.shape[1] >= 4
