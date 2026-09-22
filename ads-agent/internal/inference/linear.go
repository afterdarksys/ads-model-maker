package inference

import (
	"context"
	"crypto/sha256"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"math"
	"os"
	"strings"
)

// linearModel is the ads-linear-v1 file written by ads_ml.security.linear.
// Feature hashing must stay aligned with that module: lowercase text, wrap
// in "<" and ">", SHA-256 each character n-gram, index from the first four
// little-endian bytes, sign from whether byte 5 is even.
type linearModel struct {
	Format  string      `json:"format"`
	Task    string      `json:"task"`
	Labels  []string    `json:"labels"`
	Ngram   int         `json:"ngram"`
	Dim     int         `json:"dim"`
	Bias    []float64   `json:"bias"`
	Weights [][]float64 `json:"weights"`
}

type linearRuntime struct {
	model *linearModel
}

func newLinearRuntime(path string) (*linearRuntime, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read linear model: %w", err)
	}
	var model linearModel
	if err := json.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("parse linear model: %w", err)
	}
	if model.Format != "ads-linear-v1" {
		return nil, fmt.Errorf("unsupported linear format")
	}
	if len(model.Labels) < 2 || model.Ngram < 1 || model.Dim < 1 {
		return nil, fmt.Errorf("linear model is missing labels or dimensions")
	}
	if len(model.Bias) != len(model.Labels) || len(model.Weights) != len(model.Labels) {
		return nil, fmt.Errorf("linear model rows do not match labels")
	}
	for _, row := range model.Weights {
		if len(row) != model.Dim {
			return nil, fmt.Errorf("linear model weight row does not match dim")
		}
	}
	return &linearRuntime{model: &model}, nil
}

func (r *linearRuntime) Predict(ctx context.Context, input map[string]interface{}) (interface{}, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}
	text, _ := input["text"].(string)
	if strings.TrimSpace(text) == "" {
		return nil, fmt.Errorf("text is required")
	}
	if len([]rune(text)) > 100000 {
		return nil, fmt.Errorf("text exceeds 100000 characters")
	}

	scores := r.model.scores(text)
	probs := stableSoftmax(scores)
	best := 0
	for i := 1; i < len(probs); i++ {
		if probs[i] > probs[best] {
			best = i
		}
	}
	named := make(map[string]float64, len(r.model.Labels))
	for i, label := range r.model.Labels {
		named[label] = probs[i]
	}
	return map[string]interface{}{
		"label":         r.model.Labels[best],
		"confidence":    probs[best],
		"probabilities": named,
	}, nil
}

func (r *linearRuntime) Close() error { return nil }

func (m *linearModel) scores(text string) []float64 {
	buckets := featureBuckets(text, m.Ngram, m.Dim)
	scores := make([]float64, len(m.Labels))
	for classIndex, bias := range m.Bias {
		score := bias
		row := m.Weights[classIndex]
		for index, value := range buckets {
			score += row[index] * value
		}
		scores[classIndex] = score
	}
	return scores
}

func featureBuckets(text string, ngram, dim int) map[int]float64 {
	padded := []rune("<" + strings.ToLower(text) + ">")
	grams := make([]string, 0, len(padded))
	if len(padded) <= ngram {
		grams = append(grams, string(padded))
	} else {
		for i := 0; i <= len(padded)-ngram; i++ {
			grams = append(grams, string(padded[i:i+ngram]))
		}
	}
	buckets := make(map[int]float64)
	for _, gram := range grams {
		sum := sha256.Sum256([]byte(gram))
		index := int(binary.LittleEndian.Uint32(sum[:4])) % dim
		sign := 1.0
		if sum[4]%2 != 0 {
			sign = -1
		}
		buckets[index] += sign
	}
	return buckets
}

func stableSoftmax(scores []float64) []float64 {
	peak := scores[0]
	for _, score := range scores[1:] {
		if score > peak {
			peak = score
		}
	}
	total := 0.0
	shifted := make([]float64, len(scores))
	for i, score := range scores {
		shifted[i] = math.Exp(score - peak)
		total += shifted[i]
	}
	for i := range shifted {
		shifted[i] /= total
	}
	return shifted
}
