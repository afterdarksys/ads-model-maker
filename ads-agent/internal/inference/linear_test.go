package inference

import (
	"context"
	"math"
	"os"
	"path/filepath"
	"testing"

	"github.com/afterdarksolutions/ads-agent/internal/config"
)

func TestLinearPredictUsesBias(t *testing.T) {
	model := &linearModel{
		Format:  "ads-linear-v1",
		Labels:  []string{"no", "yes"},
		Ngram:   3,
		Dim:     8,
		Bias:    []float64{0, 3},
		Weights: [][]float64{make([]float64, 8), make([]float64, 8)},
	}
	runtime := &linearRuntime{model: model}
	result, err := runtime.Predict(context.Background(), map[string]interface{}{"text": "hello"})
	if err != nil {
		t.Fatal(err)
	}
	got := result.(map[string]interface{})
	if got["label"] != "yes" {
		t.Fatalf("label = %v", got["label"])
	}
	confidence := got["confidence"].(float64)
	expected := math.Exp(3) / (1 + math.Exp(3))
	if math.Abs(confidence-expected) > 1e-9 {
		t.Fatalf("confidence = %v, want %v", confidence, expected)
	}
}

func TestLinearRejectsEmptyText(t *testing.T) {
	runtime := &linearRuntime{model: &linearModel{
		Format:  "ads-linear-v1",
		Labels:  []string{"no", "yes"},
		Ngram:   3,
		Dim:     4,
		Bias:    []float64{0, 0},
		Weights: [][]float64{make([]float64, 4), make([]float64, 4)},
	}}
	if _, err := runtime.Predict(context.Background(), map[string]interface{}{"text": "  "}); err == nil {
		t.Fatal("empty text was accepted")
	}
}

func TestEngineRejectsUnknownTypeAndLoadsLinear(t *testing.T) {
	dir := t.TempDir()
	engine, err := NewEngine(config.InferenceConfig{CacheSizeMB: 8, CacheTTLMin: 1})
	if err != nil {
		t.Fatal(err)
	}
	defer engine.Close()

	plain := filepath.Join(dir, "notes.txt")
	if err := os.WriteFile(plain, []byte("nope"), 0o644); err != nil {
		t.Fatal(err)
	}
	if err := engine.LoadModel(context.Background(), plain); err == nil {
		t.Fatal("text file was loaded as a model")
	}

	modelPath := filepath.Join(dir, "phish.ads.json")
	body := []byte(`{"format":"ads-linear-v1","task":"phishing_email","labels":["no","yes"],"ngram":3,"dim":4,"bias":[0,2],"weights":[[0,0,0,0],[0,0,0,0]]}`)
	if err := os.WriteFile(modelPath, body, 0o644); err != nil {
		t.Fatal(err)
	}
	if err := engine.LoadModel(context.Background(), modelPath); err != nil {
		t.Fatal(err)
	}
	result, cached, err := engine.Predict(context.Background(), "phish.ads.json", map[string]interface{}{"text": "reset the password"})
	if err != nil {
		t.Fatal(err)
	}
	if cached {
		t.Fatal("first prediction was cached")
	}
	if result.(map[string]interface{})["label"] != "yes" {
		t.Fatalf("result = %#v", result)
	}
	if _, _, err := engine.Predict(context.Background(), "", map[string]interface{}{"text": "x"}); err != ErrModelNotFound {
		t.Fatalf("empty model id error = %v", err)
	}
}
