package api

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/afterdarksolutions/ads-agent/internal/config"
	"github.com/afterdarksolutions/ads-agent/internal/inference"
	"github.com/afterdarksolutions/ads-agent/internal/storage"
)

func testServer(t *testing.T, host, apiKey, modelsDir string, maxBody int) *Server {
	t.Helper()
	store, err := storage.NewDB(filepath.Join(t.TempDir(), "db"))
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { store.Close() })
	engine, err := inference.NewEngine(config.InferenceConfig{CacheSizeMB: 8, CacheTTLMin: 1})
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { engine.Close() })
	server := NewServer(config.ServerConfig{
		Host:         host,
		Port:         1,
		ReadTimeout:  1,
		WriteTimeout: 1,
		APIKey:       apiKey,
		MaxBodyBytes: maxBody,
	}, engine, store, modelsDir)
	server.SetupRoutes()
	return server
}

func TestAuthRejectsMissingAndWrongKey(t *testing.T) {
	server := testServer(t, "127.0.0.1", "correct-key", t.TempDir(), 1<<20)

	missing := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	resp, err := server.app.Test(missing, -1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("missing key status = %d", resp.StatusCode)
	}

	wrong := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	wrong.Header.Set("Authorization", "Bearer nope")
	resp, err = server.app.Test(wrong, -1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusUnauthorized {
		t.Fatalf("wrong key status = %d", resp.StatusCode)
	}

	health := httptest.NewRequest(http.MethodGet, "/health", nil)
	resp, err = server.app.Test(health, -1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("health status = %d", resp.StatusCode)
	}

	ok := httptest.NewRequest(http.MethodGet, "/metrics", nil)
	ok.Header.Set("X-API-Key", "correct-key")
	resp, err = server.app.Test(ok, -1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("right key status = %d", resp.StatusCode)
	}
}

func TestNonLoopbackWithoutKeyIsRefused(t *testing.T) {
	server := testServer(t, "0.0.0.0", "", t.TempDir(), 1<<20)
	if err := server.validateListen(); err == nil {
		t.Fatal("non-loopback listener without a key was allowed")
	}
}

func TestLoadModelRejectsPathOutsideDirectory(t *testing.T) {
	root := t.TempDir()
	models := filepath.Join(root, "models")
	if err := os.Mkdir(models, 0o755); err != nil {
		t.Fatal(err)
	}
	outside := filepath.Join(root, "outside.ads.json")
	if err := os.WriteFile(outside, []byte(`{"format":"ads-linear-v1"}`), 0o644); err != nil {
		t.Fatal(err)
	}
	server := testServer(t, "127.0.0.1", "correct-key", models, 1<<20)
	body := strings.NewReader(`{"path":"` + outside + `"}`)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/models/outside.ads.json/load", body)
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer correct-key")
	resp, err := server.app.Test(req, -1)
	if err != nil {
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}

func TestOversizedBodyIsRejected(t *testing.T) {
	server := testServer(t, "127.0.0.1", "correct-key", t.TempDir(), 32)
	req := httptest.NewRequest(http.MethodPost, "/api/v1/inference/classify", strings.NewReader(strings.Repeat("a", 200)))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Authorization", "Bearer correct-key")
	resp, err := server.app.Test(req, -1)
	if err != nil {
		if strings.Contains(err.Error(), "body size exceeds") {
			return
		}
		t.Fatal(err)
	}
	if resp.StatusCode != http.StatusRequestEntityTooLarge {
		t.Fatalf("status = %d", resp.StatusCode)
	}
}
