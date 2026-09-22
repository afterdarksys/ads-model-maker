// Package inference handles model loading and inference.
package inference

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/allegro/bigcache/v3"
	"github.com/rs/zerolog/log"

	"github.com/afterdarksolutions/ads-agent/internal/config"
)

var (
	ErrModelNotFound  = errors.New("model not found")
	ErrModelNotLoaded = errors.New("model not loaded")
	ErrInference      = errors.New("inference failed")
)

// Model represents a loaded model.
type Model struct {
	ID          string    `json:"id"`
	Name        string    `json:"name"`
	Type        string    `json:"type"` // classifier, embedder, generator
	Path        string    `json:"path"`
	LoadedAt    time.Time `json:"loaded_at"`
	Predictions int64     `json:"predictions"`
	runtime     ModelRuntime
}

// ModelRuntime interface for different model backends.
type ModelRuntime interface {
	Predict(ctx context.Context, input map[string]interface{}) (interface{}, error)
	Close() error
}

// Engine manages model loading and inference.
type Engine struct {
	config config.InferenceConfig
	models map[string]*Model
	cache  *bigcache.BigCache
	mu     sync.RWMutex

	// Stats. These move on the request path, so they are atomic.
	totalPredictions atomic.Int64
	cacheHits        atomic.Int64
	cacheMisses      atomic.Int64
}

// NewEngine creates a new inference engine.
func NewEngine(cfg config.InferenceConfig) (*Engine, error) {
	// Initialize cache
	cacheConfig := bigcache.Config{
		Shards:             1024,
		LifeWindow:         time.Duration(cfg.CacheTTLMin) * time.Minute,
		CleanWindow:        1 * time.Minute,
		MaxEntriesInWindow: 1000 * 10 * 60,
		MaxEntrySize:       512 * 1024, // 512KB per entry
		HardMaxCacheSize:   cfg.CacheSizeMB,
		Verbose:            false,
	}

	cache, err := bigcache.New(context.Background(), cacheConfig)
	if err != nil {
		return nil, fmt.Errorf("failed to create cache: %w", err)
	}

	return &Engine{
		config: cfg,
		models: make(map[string]*Model),
		cache:  cache,
	}, nil
}

// Close shuts down the engine.
func (e *Engine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	for _, model := range e.models {
		if model.runtime != nil {
			model.runtime.Close()
		}
	}

	return e.cache.Close()
}

// LoadModel loads a model from disk.
func (e *Engine) LoadModel(ctx context.Context, path string) error {
	info, err := os.Stat(path)
	if err != nil {
		return fmt.Errorf("model file: %w", err)
	}
	if info.IsDir() {
		return fmt.Errorf("model path is a directory")
	}

	modelType, err := detectModelType(path)
	if err != nil {
		return err
	}

	var runtime ModelRuntime
	switch modelType {
	case "onnx":
		runtime, err = newONNXRuntime(path, e.config)
	case "golearn":
		runtime, err = newGoLearnRuntime(path)
	case "linear":
		runtime, err = newLinearRuntime(path)
	default:
		return fmt.Errorf("unsupported model type: %s", modelType)
	}
	if err != nil {
		return fmt.Errorf("failed to load model: %w", err)
	}

	modelID := filepath.Base(path)
	e.mu.Lock()
	defer e.mu.Unlock()
	if previous, ok := e.models[modelID]; ok && previous.runtime != nil {
		_ = previous.runtime.Close()
	}
	e.models[modelID] = &Model{
		ID:       modelID,
		Name:     modelID,
		Type:     modelType,
		Path:     path,
		LoadedAt: time.Now(),
		runtime:  runtime,
	}

	log.Info().Str("model_id", modelID).Str("type", modelType).Msg("Model loaded")
	return nil
}

// UnloadModel removes a model from memory.
func (e *Engine) UnloadModel(modelID string) error {
	e.mu.Lock()
	defer e.mu.Unlock()

	model, ok := e.models[modelID]
	if !ok {
		return ErrModelNotFound
	}

	if model.runtime != nil {
		model.runtime.Close()
	}

	delete(e.models, modelID)
	log.Info().Str("model_id", modelID).Msg("Model unloaded")
	return nil
}

// GetModel returns model info.
func (e *Engine) GetModel(modelID string) (*Model, error) {
	e.mu.RLock()
	defer e.mu.RUnlock()

	model, ok := e.models[modelID]
	if !ok {
		return nil, ErrModelNotFound
	}
	return model, nil
}

// ListModels returns all loaded models.
func (e *Engine) ListModels() []*Model {
	e.mu.RLock()
	defer e.mu.RUnlock()

	models := make([]*Model, 0, len(e.models))
	for _, m := range e.models {
		models = append(models, m)
	}
	return models
}

// LoadedModelCount returns number of loaded models.
func (e *Engine) LoadedModelCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.models)
}

// Predict runs inference on a model.
func (e *Engine) Predict(ctx context.Context, modelID string, input map[string]interface{}) (interface{}, bool, error) {
	if modelID == "" {
		return nil, false, ErrModelNotFound
	}

	cacheKey := makeCacheKey(modelID, input)
	if cached, err := e.cache.Get(cacheKey); err == nil {
		e.cacheHits.Add(1)
		var result interface{}
		if err := json.Unmarshal(cached, &result); err != nil {
			return nil, false, fmt.Errorf("cached prediction is unreadable")
		}
		return result, true, nil
	}
	e.cacheMisses.Add(1)

	e.mu.RLock()
	model, ok := e.models[modelID]
	var runtime ModelRuntime
	if ok && model != nil {
		runtime = model.runtime
	}
	e.mu.RUnlock()

	if !ok || runtime == nil {
		return nil, false, ErrModelNotFound
	}

	result, err := runtime.Predict(ctx, input)
	if err != nil {
		return nil, false, fmt.Errorf("%w: %v", ErrInference, err)
	}

	e.mu.Lock()
	if current, stillLoaded := e.models[modelID]; stillLoaded {
		current.Predictions++
	}
	e.mu.Unlock()
	e.totalPredictions.Add(1)

	// Cache result
	if data, err := json.Marshal(result); err == nil {
		e.cache.Set(cacheKey, data)
	}

	return result, false, nil
}

// Classify runs classification inference.
func (e *Engine) Classify(ctx context.Context, modelID string, texts []string) ([]map[string]interface{}, error) {
	results := make([]map[string]interface{}, len(texts))

	for i, text := range texts {
		input := map[string]interface{}{"text": text}
		result, _, err := e.Predict(ctx, modelID, input)
		if err != nil {
			return nil, err
		}

		if r, ok := result.(map[string]interface{}); ok {
			results[i] = r
		} else {
			results[i] = map[string]interface{}{"result": result}
		}
	}

	return results, nil
}

// Embed generates embeddings.
func (e *Engine) Embed(ctx context.Context, modelID string, texts []string, normalize bool) ([][]float64, error) {
	embeddings := make([][]float64, len(texts))

	for i, text := range texts {
		input := map[string]interface{}{
			"text":      text,
			"normalize": normalize,
		}
		result, _, err := e.Predict(ctx, modelID, input)
		if err != nil {
			return nil, err
		}

		if emb, ok := result.([]float64); ok {
			embeddings[i] = emb
		} else if emb, ok := result.([]interface{}); ok {
			embeddings[i] = make([]float64, len(emb))
			for j, v := range emb {
				if f, ok := v.(float64); ok {
					embeddings[i][j] = f
				}
			}
		}
	}

	return embeddings, nil
}

// Generate produces text.
func (e *Engine) Generate(ctx context.Context, modelID string, prompt string, maxTokens int, temperature float64) (string, int, error) {
	input := map[string]interface{}{
		"prompt":      prompt,
		"max_tokens":  maxTokens,
		"temperature": temperature,
	}

	result, _, err := e.Predict(ctx, modelID, input)
	if err != nil {
		return "", 0, err
	}

	if r, ok := result.(map[string]interface{}); ok {
		text, _ := r["text"].(string)
		tokens, _ := r["tokens"].(float64)
		return text, int(tokens), nil
	}

	return fmt.Sprintf("%v", result), 0, nil
}

// Stats returns engine statistics.
func (e *Engine) Stats() map[string]interface{} {
	e.mu.RLock()
	defer e.mu.RUnlock()

	return map[string]interface{}{
		"models_loaded":     len(e.models),
		"total_predictions": e.totalPredictions.Load(),
		"cache_hits":        e.cacheHits.Load(),
		"cache_misses":      e.cacheMisses.Load(),
		"cache_entries":     e.cache.Len(),
	}
}

// ============================================================================
// Helpers
// ============================================================================

func detectModelType(path string) (string, error) {
	lower := strings.ToLower(path)
	switch {
	case strings.HasSuffix(lower, ".onnx"):
		return "onnx", nil
	case strings.HasSuffix(lower, ".ads.json"):
		return "linear", nil
	case strings.HasSuffix(lower, ".gob"), strings.HasSuffix(lower, ".golearn"):
		return "golearn", nil
	default:
		return "", fmt.Errorf("unsupported model file type")
	}
}

func makeCacheKey(modelID string, input map[string]interface{}) string {
	data, _ := json.Marshal(input)
	return fmt.Sprintf("%s:%x", modelID, data)
}

// ============================================================================
// ONNX Runtime (placeholder - needs onnxruntime_go)
// ============================================================================

type onnxRuntime struct {
	path string
}

func newONNXRuntime(path string, cfg config.InferenceConfig) (*onnxRuntime, error) {
	// TODO: Initialize ONNX runtime
	log.Info().Str("path", path).Msg("Loading ONNX model")
	return &onnxRuntime{path: path}, nil
}

func (r *onnxRuntime) Predict(ctx context.Context, input map[string]interface{}) (interface{}, error) {
	// TODO: Implement ONNX inference
	return map[string]interface{}{
		"label":      "placeholder",
		"confidence": 0.95,
	}, nil
}

func (r *onnxRuntime) Close() error {
	return nil
}

// ============================================================================
// GoLearn Runtime (placeholder)
// ============================================================================

type goLearnRuntime struct {
	path string
}

func newGoLearnRuntime(path string) (*goLearnRuntime, error) {
	log.Info().Str("path", path).Msg("Loading GoLearn model")
	return &goLearnRuntime{path: path}, nil
}

func (r *goLearnRuntime) Predict(ctx context.Context, input map[string]interface{}) (interface{}, error) {
	// TODO: Implement GoLearn inference
	return map[string]interface{}{
		"label":      "placeholder",
		"confidence": 0.90,
	}, nil
}

func (r *goLearnRuntime) Close() error {
	return nil
}
