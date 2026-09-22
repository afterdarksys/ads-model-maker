// Package api implements the HTTP API using Fiber.
package api

import (
	"crypto/rand"
	"crypto/sha256"
	"crypto/subtle"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"path/filepath"
	"strings"
	"time"

	"github.com/gofiber/fiber/v2"
	"github.com/gofiber/fiber/v2/middleware/compress"
	"github.com/gofiber/fiber/v2/middleware/cors"
	"github.com/gofiber/fiber/v2/middleware/limiter"
	"github.com/gofiber/fiber/v2/middleware/recover"
	"github.com/gofiber/websocket/v2"
	"github.com/rs/zerolog/log"

	"github.com/afterdarksolutions/ads-agent/internal/config"
	"github.com/afterdarksolutions/ads-agent/internal/inference"
	"github.com/afterdarksolutions/ads-agent/internal/storage"
)

// Server is the HTTP API server.
type Server struct {
	app       *fiber.App
	config    config.ServerConfig
	engine    *inference.Engine
	storage   *storage.DB
	modelsDir string
}

// NewServer creates a new API server.
func NewServer(cfg config.ServerConfig, engine *inference.Engine, store *storage.DB, modelsDir string) *Server {
	if cfg.MaxBodyBytes <= 0 {
		cfg.MaxBodyBytes = 1 << 20
	}
	app := fiber.New(fiber.Config{
		// Performance tuning
		ServerHeader:          "ADS-Agent",
		StrictRouting:         true,
		CaseSensitive:         true,
		DisableStartupMessage: true,
		ReduceMemoryUsage:     true,
		BodyLimit:             cfg.MaxBodyBytes,

		JSONEncoder: json.Marshal,
		JSONDecoder: json.Unmarshal,

		// Timeouts
		ReadTimeout:  time.Duration(cfg.ReadTimeout) * time.Second,
		WriteTimeout: time.Duration(cfg.WriteTimeout) * time.Second,
		IdleTimeout:  120 * time.Second,

		// Error handling
		ErrorHandler: errorHandler,
	})

	return &Server{
		app:       app,
		config:    cfg,
		engine:    engine,
		storage:   store,
		modelsDir: modelsDir,
	}
}

// SetupRoutes configures all API routes.
func (s *Server) SetupRoutes() {
	// Global middleware
	s.app.Use(recover.New())
	s.app.Use(compress.New(compress.Config{
		Level: compress.LevelBestSpeed,
	}))
	s.app.Use(cors.New(cors.Config{
		AllowOrigins: "*",
		AllowMethods: "GET,POST,DELETE,OPTIONS",
		AllowHeaders: "Authorization,Content-Type,X-API-Key",
	}))
	s.app.Use(s.authMiddleware())
	s.app.Use(requestLogger())

	// Health endpoints (no auth)
	s.app.Get("/health", s.healthHandler)
	s.app.Get("/ready", s.readyHandler)

	// Metrics
	s.app.Get("/metrics", s.metricsHandler)

	// API v1
	v1 := s.app.Group("/api/v1")

	// Rate limiting
	v1.Use(limiter.New(limiter.Config{
		Max:        100,
		Expiration: 1 * time.Minute,
		KeyGenerator: func(c *fiber.Ctx) string {
			// A caller-supplied key would let someone rotate the bucket
			// and skip the limit. The address is the limiter key.
			return c.IP()
		},
	}))

	// Inference endpoints
	inference := v1.Group("/inference")
	inference.Post("/predict", s.predictHandler)
	inference.Post("/classify", s.classifyHandler)
	inference.Post("/embed", s.embedHandler)
	inference.Post("/generate", s.generateHandler)

	// Model management
	models := v1.Group("/models")
	models.Get("/", s.listModelsHandler)
	models.Get("/:id", s.getModelHandler)
	models.Post("/:id/load", s.loadModelHandler)
	models.Post("/:id/unload", s.unloadModelHandler)

	// MCP endpoints
	mcp := s.app.Group("/mcp")
	mcp.Get("/v1/tools", s.mcpListToolsHandler)
	mcp.Post("/v1/tools/:name/call", s.mcpCallToolHandler)

	// MCP WebSocket
	s.app.Get("/mcp/ws", websocket.New(s.mcpWebSocketHandler))
}

// Start begins listening for requests.
func (s *Server) Start() error {
	if err := s.validateListen(); err != nil {
		return err
	}
	addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
	log.Info().Str("addr", addr).Msg("Server listening")
	return s.app.Listen(addr)
}

// Shutdown gracefully stops the server.
func (s *Server) Shutdown() error {
	return s.app.ShutdownWithTimeout(10 * time.Second)
}

// ============================================================================
// Health Handlers
// ============================================================================

func (s *Server) healthHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{
		"status":  "healthy",
		"version": "0.1.0",
	})
}

func (s *Server) readyHandler(c *fiber.Ctx) error {
	modelsLoaded := s.engine.LoadedModelCount()
	return c.JSON(fiber.Map{
		"status":        "ready",
		"models_loaded": modelsLoaded,
	})
}

func (s *Server) metricsHandler(c *fiber.Ctx) error {
	stats := s.engine.Stats()
	return c.JSON(stats)
}

// ============================================================================
// Inference Handlers
// ============================================================================

// PredictRequest for generic prediction.
type PredictRequest struct {
	ModelID string                 `json:"model_id"`
	Input   map[string]interface{} `json:"input"`
	Options map[string]interface{} `json:"options,omitempty"`
}

// PredictResponse from inference.
type PredictResponse struct {
	ID        string      `json:"id"`
	ModelID   string      `json:"model_id"`
	Result    interface{} `json:"result"`
	LatencyMs int64       `json:"latency_ms"`
	Cached    bool        `json:"cached"`
}

func (s *Server) predictHandler(c *fiber.Ctx) error {
	var req PredictRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if req.ModelID == "" {
		return fiber.NewError(fiber.StatusBadRequest, "model_id is required")
	}

	ctx := c.Context()
	start := time.Now()

	result, cached, err := s.engine.Predict(ctx, req.ModelID, req.Input)
	if err != nil {
		return handleInferenceError(err)
	}

	return c.JSON(PredictResponse{
		ID:        generateID(),
		ModelID:   req.ModelID,
		Result:    result,
		LatencyMs: time.Since(start).Milliseconds(),
		Cached:    cached,
	})
}

// ClassifyRequest for classification.
type ClassifyRequest struct {
	ModelID string   `json:"model_id"`
	Texts   []string `json:"texts"`
}

// ClassifyResult for a single text.
type ClassifyResult struct {
	Text       string             `json:"text"`
	Label      string             `json:"label"`
	Confidence float64            `json:"confidence"`
	Probs      map[string]float64 `json:"probabilities,omitempty"`
}

func (s *Server) classifyHandler(c *fiber.Ctx) error {
	var req ClassifyRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if len(req.Texts) == 0 {
		return fiber.NewError(fiber.StatusBadRequest, "texts array is required")
	}

	ctx := c.Context()
	start := time.Now()

	results, err := s.engine.Classify(ctx, req.ModelID, req.Texts)
	if err != nil {
		return handleInferenceError(err)
	}

	return c.JSON(fiber.Map{
		"id":         generateID(),
		"model_id":   req.ModelID,
		"results":    results,
		"latency_ms": time.Since(start).Milliseconds(),
	})
}

// EmbedRequest for embeddings.
type EmbedRequest struct {
	ModelID   string   `json:"model_id"`
	Texts     []string `json:"texts"`
	Normalize bool     `json:"normalize"`
}

func (s *Server) embedHandler(c *fiber.Ctx) error {
	var req EmbedRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if len(req.Texts) == 0 {
		return fiber.NewError(fiber.StatusBadRequest, "texts array is required")
	}

	ctx := c.Context()
	start := time.Now()

	embeddings, err := s.engine.Embed(ctx, req.ModelID, req.Texts, req.Normalize)
	if err != nil {
		return handleInferenceError(err)
	}

	return c.JSON(fiber.Map{
		"id":         generateID(),
		"model_id":   req.ModelID,
		"embeddings": embeddings,
		"dimensions": embeddingWidth(embeddings),
		"latency_ms": time.Since(start).Milliseconds(),
	})
}

// GenerateRequest for text generation.
type GenerateRequest struct {
	ModelID     string  `json:"model_id"`
	Prompt      string  `json:"prompt"`
	MaxTokens   int     `json:"max_tokens"`
	Temperature float64 `json:"temperature"`
}

func (s *Server) generateHandler(c *fiber.Ctx) error {
	var req GenerateRequest
	if err := c.BodyParser(&req); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	if req.Prompt == "" {
		return fiber.NewError(fiber.StatusBadRequest, "prompt is required")
	}

	// Defaults
	if req.MaxTokens == 0 {
		req.MaxTokens = 256
	}
	if req.Temperature == 0 {
		req.Temperature = 0.8
	}

	ctx := c.Context()
	start := time.Now()

	text, tokens, err := s.engine.Generate(ctx, req.ModelID, req.Prompt, req.MaxTokens, req.Temperature)
	if err != nil {
		return handleInferenceError(err)
	}

	return c.JSON(fiber.Map{
		"id":               generateID(),
		"model_id":         req.ModelID,
		"text":             text,
		"tokens_generated": tokens,
		"latency_ms":       time.Since(start).Milliseconds(),
	})
}

// ============================================================================
// Model Handlers
// ============================================================================

func (s *Server) listModelsHandler(c *fiber.Ctx) error {
	models := s.engine.ListModels()
	return c.JSON(fiber.Map{
		"models": models,
		"total":  len(models),
	})
}

func (s *Server) getModelHandler(c *fiber.Ctx) error {
	id := c.Params("id")
	model, err := s.engine.GetModel(id)
	if err != nil {
		return fiber.NewError(fiber.StatusNotFound, "Model not found")
	}
	return c.JSON(model)
}

func (s *Server) loadModelHandler(c *fiber.Ctx) error {
	id := c.Params("id")

	var body struct {
		Path string `json:"path"`
	}
	if err := c.BodyParser(&body); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	ctx := c.Context()
	modelPath, err := s.modelPath(id, body.Path)
	if err != nil {
		return err
	}

	if err := s.engine.LoadModel(ctx, modelPath); err != nil {
		log.Warn().Err(err).Str("model_id", id).Msg("model load failed")
		return fiber.NewError(fiber.StatusBadRequest, "failed to load model")
	}

	return c.JSON(fiber.Map{
		"status":   "loaded",
		"model_id": id,
	})
}

func (s *Server) unloadModelHandler(c *fiber.Ctx) error {
	id := c.Params("id")

	if err := s.engine.UnloadModel(id); err != nil {
		return fiber.NewError(fiber.StatusNotFound, err.Error())
	}

	return c.JSON(fiber.Map{
		"status":   "unloaded",
		"model_id": id,
	})
}

// ============================================================================
// MCP Handlers
// ============================================================================

func (s *Server) mcpListToolsHandler(c *fiber.Ctx) error {
	return c.JSON(fiber.Map{"tools": mcpTools()})
}

func mcpTools() []fiber.Map {
	return []fiber.Map{
		{
			"name":        "classify_text",
			"description": "Classify text into categories using loaded model",
			"inputSchema": fiber.Map{
				"type": "object",
				"properties": fiber.Map{
					"text":     fiber.Map{"type": "string"},
					"model_id": fiber.Map{"type": "string"},
				},
				"required": []string{"text"},
			},
		},
		{
			"name":        "embed_text",
			"description": "Generate embeddings for semantic search",
			"inputSchema": fiber.Map{
				"type": "object",
				"properties": fiber.Map{
					"text":     fiber.Map{"type": "string"},
					"model_id": fiber.Map{"type": "string"},
				},
				"required": []string{"text"},
			},
		},
		{
			"name":        "generate_text",
			"description": "Generate text continuation",
			"inputSchema": fiber.Map{
				"type": "object",
				"properties": fiber.Map{
					"prompt":      fiber.Map{"type": "string"},
					"model_id":    fiber.Map{"type": "string"},
					"max_tokens":  fiber.Map{"type": "integer"},
					"temperature": fiber.Map{"type": "number"},
				},
				"required": []string{"prompt"},
			},
		},
	}
}

func (s *Server) mcpCallToolHandler(c *fiber.Ctx) error {
	name := c.Params("name")

	var body struct {
		Arguments map[string]interface{} `json:"arguments"`
	}
	if err := c.BodyParser(&body); err != nil {
		return fiber.NewError(fiber.StatusBadRequest, "Invalid request body")
	}

	ctx := c.Context()

	switch name {
	case "classify_text":
		text, _ := body.Arguments["text"].(string)
		modelID, _ := body.Arguments["model_id"].(string)
		if strings.TrimSpace(text) == "" || modelID == "" {
			return fiber.NewError(fiber.StatusBadRequest, "text and model_id are required")
		}

		results, err := s.engine.Classify(ctx, modelID, []string{text})
		if err != nil {
			return handleInferenceError(err)
		}
		if len(results) == 0 {
			return fiber.NewError(fiber.StatusInternalServerError, "classification returned no result")
		}

		return c.JSON(fiber.Map{
			"content": []fiber.Map{{
				"type": "text",
				"text": fmt.Sprintf("Classification: %v", results[0]),
			}},
		})

	case "embed_text":
		text, _ := body.Arguments["text"].(string)
		modelID, _ := body.Arguments["model_id"].(string)

		embeddings, err := s.engine.Embed(ctx, modelID, []string{text}, true)
		if err != nil {
			return fiber.NewError(fiber.StatusInternalServerError, err.Error())
		}

		return c.JSON(fiber.Map{
			"content": []fiber.Map{{
				"type": "text",
				"text": fmt.Sprintf("Embedding dimensions: %d", len(embeddings[0])),
			}},
		})

	case "generate_text":
		prompt, _ := body.Arguments["prompt"].(string)
		modelID, _ := body.Arguments["model_id"].(string)
		maxTokens := 256
		if mt, ok := body.Arguments["max_tokens"].(float64); ok {
			maxTokens = int(mt)
		}

		text, _, err := s.engine.Generate(ctx, modelID, prompt, maxTokens, 0.8)
		if err != nil {
			return fiber.NewError(fiber.StatusInternalServerError, err.Error())
		}

		return c.JSON(fiber.Map{
			"content": []fiber.Map{{
				"type": "text",
				"text": text,
			}},
		})

	default:
		return fiber.NewError(fiber.StatusNotFound, "Tool not found")
	}
}

func (s *Server) mcpWebSocketHandler(c *websocket.Conn) {
	for {
		_, msg, err := c.ReadMessage()
		if err != nil {
			break
		}

		var request map[string]interface{}
		if err := json.Unmarshal(msg, &request); err != nil {
			continue
		}

		// Handle MCP JSON-RPC
		method, _ := request["method"].(string)
		id := request["id"]

		var response interface{}

		switch method {
		case "tools/list":
			response = fiber.Map{
				"jsonrpc": "2.0",
				"id":      id,
				"result":  fiber.Map{"tools": mcpTools()},
			}
		default:
			response = fiber.Map{
				"jsonrpc": "2.0",
				"id":      id,
				"error":   fiber.Map{"code": -32601, "message": "Method not found"},
			}
		}

		data, _ := json.Marshal(response)
		c.WriteMessage(websocket.TextMessage, data)
	}
}

// ============================================================================
// Helpers
// ============================================================================

func errorHandler(c *fiber.Ctx, err error) error {
	code := fiber.StatusInternalServerError
	message := "Internal server error"

	if e, ok := err.(*fiber.Error); ok {
		code = e.Code
		message = e.Message
	}

	return c.Status(code).JSON(fiber.Map{
		"error": fiber.Map{
			"code":    code,
			"message": message,
		},
	})
}

func handleInferenceError(err error) error {
	if errors.Is(err, inference.ErrModelNotFound) || errors.Is(err, inference.ErrModelNotLoaded) {
		return fiber.NewError(fiber.StatusNotFound, "Model not found")
	}
	if errors.Is(err, inference.ErrInference) {
		return fiber.NewError(fiber.StatusBadRequest, "inference failed")
	}
	return fiber.NewError(fiber.StatusInternalServerError, "inference failed")
}

func requestLogger() fiber.Handler {
	return func(c *fiber.Ctx) error {
		start := time.Now()
		err := c.Next()
		status := c.Response().StatusCode()
		var fiberErr *fiber.Error
		if errors.As(err, &fiberErr) {
			status = fiberErr.Code
		}

		log.Debug().
			Str("method", c.Method()).
			Str("path", c.Path()).
			Int("status", status).
			Dur("latency", time.Since(start)).
			Msg("request")

		return err
	}
}

func generateID() string {
	buf := make([]byte, 16)
	if _, err := rand.Read(buf); err != nil {
		return fmt.Sprintf("req_%d", time.Now().UnixNano())
	}
	return hex.EncodeToString(buf)
}

func embeddingWidth(rows [][]float64) int {
	if len(rows) == 0 || rows[0] == nil {
		return 0
	}
	return len(rows[0])
}

// Threats: blocks anonymous use of inference, model loading, metrics, and MCP
// when an API key is set. A missing key is allowed only on a loopback bind.
// This does not stop someone who already has the key, and it does not
// encrypt the request body.
func (s *Server) authMiddleware() fiber.Handler {
	return func(c *fiber.Ctx) error {
		if c.Path() == "/health" || c.Path() == "/ready" {
			return c.Next()
		}
		if s.config.APIKey == "" {
			if loopbackHost(s.config.Host) {
				return c.Next()
			}
			return fiber.NewError(fiber.StatusUnauthorized, "unauthorized")
		}
		presented := bearerToken(c.Get("Authorization"))
		if presented == "" {
			presented = c.Get("X-API-Key")
		}
		if !keysEqual(presented, s.config.APIKey) {
			return fiber.NewError(fiber.StatusUnauthorized, "unauthorized")
		}
		return c.Next()
	}
}

func (s *Server) validateListen() error {
	if s.config.APIKey == "" && !loopbackHost(s.config.Host) {
		return fmt.Errorf("refusing to listen on %s without an API key", s.config.Host)
	}
	return nil
}

func (s *Server) modelPath(id, raw string) (string, error) {
	if id == "" || strings.TrimSpace(raw) == "" {
		return "", fiber.NewError(fiber.StatusBadRequest, "model id and path are required")
	}
	if strings.Contains(id, "/") || strings.Contains(id, "\\") || id == "." || id == ".." {
		return "", fiber.NewError(fiber.StatusBadRequest, "invalid model id")
	}
	if s.modelsDir == "" {
		return "", fiber.NewError(fiber.StatusForbidden, "model directory is not configured")
	}
	base, err := filepath.Abs(s.modelsDir)
	if err != nil {
		return "", fiber.NewError(fiber.StatusInternalServerError, "model directory is not available")
	}
	abs, err := filepath.Abs(filepath.Clean(raw))
	if err != nil {
		return "", fiber.NewError(fiber.StatusBadRequest, "invalid model path")
	}
	rel, err := filepath.Rel(base, abs)
	if err != nil || rel == ".." || strings.HasPrefix(rel, ".."+string(filepath.Separator)) {
		return "", fiber.NewError(fiber.StatusBadRequest, "model path is outside the model directory")
	}
	if filepath.Base(abs) != id {
		return "", fiber.NewError(fiber.StatusBadRequest, "model id does not match the file name")
	}
	return abs, nil
}

func loopbackHost(host string) bool {
	switch strings.Trim(strings.TrimSpace(host), "[]") {
	case "127.0.0.1", "localhost", "::1":
		return true
	default:
		return false
	}
}

func bearerToken(header string) string {
	const prefix = "Bearer "
	if len(header) < len(prefix) || !strings.EqualFold(header[:len(prefix)], prefix) {
		return ""
	}
	return strings.TrimSpace(header[len(prefix):])
}

func keysEqual(got, want string) bool {
	sumGot := sha256.Sum256([]byte(got))
	sumWant := sha256.Sum256([]byte(want))
	return subtle.ConstantTimeCompare(sumGot[:], sumWant[:]) == 1
}
