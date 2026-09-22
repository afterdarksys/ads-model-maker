# Go Agent Design

## Overview

The Go agent is a lightweight, high-performance binary that runs on user machines. It handles:
- Local filesystem scanning and data collection
- Model serving and inference (local)
- MCP (Model Context Protocol) server for LLM integration
- Agent capabilities for autonomous tasks
- Secure communication with cloud backend

## Design Principles

1. **Zero Dependencies** - Single binary, no runtime requirements
2. **Blazing Fast** - Sub-millisecond response times for inference
3. **Cross-Platform** - Windows, macOS, Linux from single codebase
4. **Memory Efficient** - Low RAM footprint, suitable for edge
5. **Secure by Default** - Encrypted comms, minimal permissions

## High-Performance Stack

### Core Libraries

| Library | Purpose | Why It's Fast |
|---------|---------|---------------|
| `gofiber/fiber` | HTTP framework | Built on fasthttp, 10x net/http |
| `valyala/fasthttp` | HTTP client/server | Zero allocation, connection pooling |
| `bytedance/sonic` | JSON encode/decode | SIMD acceleration, 2-3x encoding/json |
| `rs/zerolog` | Structured logging | Zero allocation logging |
| `allegro/bigcache` | In-memory cache | Concurrent, GB-scale, no GC pressure |
| `dgraph-io/badger` | Embedded KV store | LSM-tree, SSD-optimized |
| `yalue/onnxruntime_go` | ONNX inference | Native bindings, GPU support |
| `gorgonia/gorgonia` | ML/tensor ops | Pure Go, auto-differentiation |
| `sjwhitworth/golearn` | ML algorithms | Classical ML in Go |
| `panjf2000/ants` | Goroutine pool | Bounded concurrency |

### Performance Targets

| Operation | Target Latency | Memory |
|-----------|----------------|--------|
| Health check | < 1ms | - |
| Small inference | < 10ms | < 50MB |
| Large inference | < 100ms | < 500MB |
| File scan (1000 files) | < 500ms | < 100MB |
| Model load | < 2s | Model size + 20% |

## Project Structure

```
ads-agent/
├── cmd/
│   └── agent/
│       └── main.go                 # Entry point
├── internal/
│   ├── api/
│   │   ├── server.go               # Fiber HTTP server
│   │   ├── routes.go               # Route definitions
│   │   ├── handlers/
│   │   │   ├── inference.go        # Inference endpoints
│   │   │   ├── health.go           # Health checks
│   │   │   ├── data.go             # Data collection
│   │   │   └── models.go           # Model management
│   │   └── middleware/
│   │       ├── auth.go             # API key auth
│   │       ├── ratelimit.go        # Rate limiting
│   │       └── logging.go          # Request logging
│   ├── inference/
│   │   ├── engine.go               # Inference orchestrator
│   │   ├── onnx.go                 # ONNX runtime
│   │   ├── golearn.go              # GoLearn models
│   │   ├── gorgonia.go             # Gorgonia models
│   │   └── cache.go                # Response caching
│   ├── mcp/
│   │   ├── server.go               # MCP protocol server
│   │   ├── tools.go                # Tool definitions
│   │   ├── resources.go            # Resource handlers
│   │   └── prompts.go              # Prompt templates
│   ├── agent/
│   │   ├── agent.go                # Autonomous agent
│   │   ├── planner.go              # Task planning
│   │   ├── executor.go             # Action execution
│   │   └── memory.go               # Agent memory/state
│   ├── collector/
│   │   ├── scanner.go              # File scanner
│   │   ├── filesystem.go           # FS operations
│   │   ├── uploader.go             # Chunked upload
│   │   └── filters.go              # File filters
│   ├── models/
│   │   ├── manager.go              # Model lifecycle
│   │   ├── loader.go               # Model loading
│   │   ├── registry.go             # Model registry
│   │   └── formats.go              # Format handlers
│   ├── storage/
│   │   ├── cache.go                # BigCache wrapper
│   │   ├── db.go                   # BadgerDB wrapper
│   │   └── models.go               # Model storage
│   ├── tunnel/
│   │   ├── client.go               # Backend connection
│   │   ├── tls.go                  # mTLS handling
│   │   └── reconnect.go            # Auto-reconnect
│   └── config/
│       └── config.go               # Configuration
├── pkg/
│   ├── formats/                    # File format detection
│   ├── crypto/                     # Encryption utilities
│   └── metrics/                    # Prometheus metrics
├── configs/
│   └── agent.yaml                  # Default config
├── scripts/
│   ├── build.sh                    # Cross-compilation
│   └── install.sh                  # Installer script
├── go.mod
└── go.sum
```

## Core Components

### 1. High-Performance HTTP Server

```go
// internal/api/server.go
package api

import (
    "github.com/gofiber/fiber/v2"
    "github.com/gofiber/fiber/v2/middleware/compress"
    "github.com/gofiber/fiber/v2/middleware/cors"
    "github.com/gofiber/fiber/v2/middleware/recover"
    "github.com/rs/zerolog/log"
)

type Server struct {
    app      *fiber.App
    config   *Config
    engine   *inference.Engine
    mcp      *mcp.Server
    agent    *agent.Agent
}

func NewServer(config *Config) *Server {
    app := fiber.New(fiber.Config{
        // Aggressive performance tuning
        Prefork:               false,  // Set true for multi-core
        ServerHeader:          "ADS-Agent",
        StrictRouting:         true,
        CaseSensitive:         true,
        DisableStartupMessage: true,

        // Memory optimization
        ReduceMemoryUsage:     true,

        // JSON using sonic
        JSONEncoder:           sonic.Marshal,
        JSONDecoder:           sonic.Unmarshal,

        // Connection tuning
        ReadTimeout:           5 * time.Second,
        WriteTimeout:          10 * time.Second,
        IdleTimeout:           120 * time.Second,
    })

    // Middleware stack (order matters for performance)
    app.Use(recover.New())
    app.Use(compress.New(compress.Config{
        Level: compress.LevelBestSpeed,
    }))
    app.Use(cors.New())

    return &Server{
        app:    app,
        config: config,
    }
}

func (s *Server) SetupRoutes() {
    // Health (no auth)
    s.app.Get("/health", s.handlers.Health)
    s.app.Get("/ready", s.handlers.Ready)

    // API v1 (with auth)
    v1 := s.app.Group("/api/v1", s.middleware.Auth)

    // Inference endpoints
    inference := v1.Group("/inference")
    inference.Post("/predict", s.handlers.Predict)
    inference.Post("/embed", s.handlers.Embed)
    inference.Post("/classify", s.handlers.Classify)
    inference.Post("/qa", s.handlers.QuestionAnswer)

    // Model management
    models := v1.Group("/models")
    models.Get("/", s.handlers.ListModels)
    models.Get("/:id", s.handlers.GetModel)
    models.Post("/:id/load", s.handlers.LoadModel)
    models.Post("/:id/unload", s.handlers.UnloadModel)

    // Data collection
    data := v1.Group("/data")
    data.Post("/scan", s.handlers.ScanDirectory)
    data.Post("/upload", s.handlers.InitiateUpload)
    data.Get("/status/:id", s.handlers.UploadStatus)

    // Agent endpoints
    agent := v1.Group("/agent")
    agent.Post("/task", s.handlers.ExecuteTask)
    agent.Get("/status/:id", s.handlers.TaskStatus)
    agent.Post("/cancel/:id", s.handlers.CancelTask)
}

func (s *Server) Start() error {
    addr := fmt.Sprintf("%s:%d", s.config.Host, s.config.Port)
    log.Info().Str("addr", addr).Msg("Starting ADS Agent")
    return s.app.Listen(addr)
}
```

### 2. Inference Engine

```go
// internal/inference/engine.go
package inference

import (
    "sync"

    "github.com/allegro/bigcache/v3"
    ort "github.com/yalue/onnxruntime_go"
)

type Engine struct {
    models    map[string]Model
    cache     *bigcache.BigCache
    mu        sync.RWMutex

    // Model-specific engines
    onnx      *ONNXRuntime
    golearn   *GoLearnRuntime
    gorgonia  *GorgoniaRuntime
}

func NewEngine(config *Config) (*Engine, error) {
    // Initialize cache (1GB, 10 minute TTL)
    cache, err := bigcache.New(context.Background(), bigcache.Config{
        Shards:             1024,
        LifeWindow:         10 * time.Minute,
        CleanWindow:        1 * time.Minute,
        MaxEntriesInWindow: 1000 * 10 * 60,
        MaxEntrySize:       500 * 1024, // 500KB max entry
        HardMaxCacheSize:   1024,       // 1GB
        Verbose:            false,
    })
    if err != nil {
        return nil, fmt.Errorf("failed to create cache: %w", err)
    }

    // Initialize ONNX runtime
    ort.SetSharedLibraryPath(config.ONNXLibPath)
    if err := ort.InitializeEnvironment(); err != nil {
        return nil, fmt.Errorf("failed to init ONNX: %w", err)
    }

    return &Engine{
        models: make(map[string]Model),
        cache:  cache,
        onnx:   NewONNXRuntime(config),
    }, nil
}

func (e *Engine) Predict(ctx context.Context, req *PredictRequest) (*PredictResponse, error) {
    // Check cache first
    cacheKey := req.CacheKey()
    if cached, err := e.cache.Get(cacheKey); err == nil {
        var resp PredictResponse
        if err := sonic.Unmarshal(cached, &resp); err == nil {
            resp.Cached = true
            return &resp, nil
        }
    }

    // Get model
    e.mu.RLock()
    model, ok := e.models[req.ModelID]
    e.mu.RUnlock()

    if !ok {
        return nil, ErrModelNotLoaded
    }

    // Run inference
    start := time.Now()
    result, err := model.Predict(ctx, req.Input)
    if err != nil {
        return nil, fmt.Errorf("inference failed: %w", err)
    }

    resp := &PredictResponse{
        Result:    result,
        LatencyMs: time.Since(start).Milliseconds(),
        ModelID:   req.ModelID,
    }

    // Cache result
    if data, err := sonic.Marshal(resp); err == nil {
        e.cache.Set(cacheKey, data)
    }

    return resp, nil
}

func (e *Engine) LoadModel(ctx context.Context, modelPath string, modelType string) error {
    e.mu.Lock()
    defer e.mu.Unlock()

    var model Model
    var err error

    switch modelType {
    case "onnx":
        model, err = e.onnx.Load(modelPath)
    case "golearn":
        model, err = e.golearn.Load(modelPath)
    case "gorgonia":
        model, err = e.gorgonia.Load(modelPath)
    default:
        return fmt.Errorf("unsupported model type: %s", modelType)
    }

    if err != nil {
        return err
    }

    e.models[model.ID()] = model
    return nil
}
```

### 3. ONNX Runtime

```go
// internal/inference/onnx.go
package inference

import (
    ort "github.com/yalue/onnxruntime_go"
)

type ONNXRuntime struct {
    sessions map[string]*ort.AdvancedSession
    config   *Config
}

type ONNXModel struct {
    id       string
    session  *ort.AdvancedSession
    inputs   []ort.InputOutputInfo
    outputs  []ort.InputOutputInfo
}

func (r *ONNXRuntime) Load(modelPath string) (*ONNXModel, error) {
    // Create session options
    opts, err := ort.NewSessionOptions()
    if err != nil {
        return nil, err
    }
    defer opts.Destroy()

    // Enable optimizations
    opts.SetIntraOpNumThreads(runtime.NumCPU())
    opts.SetGraphOptimizationLevel(ort.GraphOptLevelAll)

    // GPU acceleration if available
    if r.config.EnableGPU {
        cudaOpts, _ := ort.NewCUDAProviderOptions()
        opts.AppendExecutionProviderCUDA(cudaOpts)
    }

    // Create session
    session, err := ort.NewAdvancedSession(
        modelPath,
        []string{"input_ids", "attention_mask"},
        []string{"logits"},
        opts,
    )
    if err != nil {
        return nil, fmt.Errorf("failed to create session: %w", err)
    }

    return &ONNXModel{
        id:      filepath.Base(modelPath),
        session: session,
    }, nil
}

func (m *ONNXModel) Predict(ctx context.Context, inputs map[string]interface{}) (interface{}, error) {
    // Convert inputs to ONNX tensors
    inputTensors, err := m.prepareInputs(inputs)
    if err != nil {
        return nil, err
    }
    defer m.destroyTensors(inputTensors)

    // Allocate output tensors
    outputTensors, err := m.allocateOutputs()
    if err != nil {
        return nil, err
    }
    defer m.destroyTensors(outputTensors)

    // Run inference
    if err := m.session.Run(); err != nil {
        return nil, fmt.Errorf("inference run failed: %w", err)
    }

    // Extract results
    return m.extractOutputs(outputTensors)
}
```

### 4. GoLearn Integration

```go
// internal/inference/golearn.go
package inference

import (
    "github.com/sjwhitworth/golearn/base"
    "github.com/sjwhitworth/golearn/ensemble"
    "github.com/sjwhitworth/golearn/knn"
    "github.com/sjwhitworth/golearn/linear_models"
    "github.com/sjwhitworth/golearn/trees"
)

type GoLearnRuntime struct {
    models map[string]base.Classifier
}

func NewGoLearnRuntime() *GoLearnRuntime {
    return &GoLearnRuntime{
        models: make(map[string]base.Classifier),
    }
}

func (r *GoLearnRuntime) Load(modelPath string) (*GoLearnModel, error) {
    // Load serialized model
    data, err := os.ReadFile(modelPath)
    if err != nil {
        return nil, err
    }

    var meta ModelMeta
    if err := sonic.Unmarshal(data[:metaSize], &meta); err != nil {
        return nil, err
    }

    var classifier base.Classifier

    switch meta.Type {
    case "random_forest":
        classifier = new(ensemble.RandomForest)
    case "knn":
        classifier = knn.NewKnnClassifier("euclidean", "linear", 5)
    case "decision_tree":
        classifier = trees.NewID3DecisionTree(0.6)
    case "logistic_regression":
        classifier = linear_models.NewLogisticRegression("l2", 1.0, 1e-4)
    default:
        return nil, fmt.Errorf("unknown model type: %s", meta.Type)
    }

    // Deserialize model weights
    if err := gob.NewDecoder(bytes.NewReader(data[metaSize:])).Decode(classifier); err != nil {
        return nil, err
    }

    return &GoLearnModel{
        id:         meta.ID,
        classifier: classifier,
        meta:       meta,
    }, nil
}

func (m *GoLearnModel) Predict(ctx context.Context, input map[string]interface{}) (interface{}, error) {
    // Convert input to golearn format
    instance := m.toInstance(input)

    // Get prediction
    predictions, err := m.classifier.Predict(instance)
    if err != nil {
        return nil, err
    }

    return m.formatPredictions(predictions), nil
}
```

### 5. MCP (Model Context Protocol) Server

```go
// internal/mcp/server.go
package mcp

import (
    "github.com/gofiber/fiber/v2"
    "github.com/gofiber/websocket/v2"
)

// MCP Server allows LLMs (Claude, etc.) to use our models as tools
type Server struct {
    engine   *inference.Engine
    tools    map[string]*Tool
    prompts  map[string]*Prompt
}

type Tool struct {
    Name        string           `json:"name"`
    Description string           `json:"description"`
    InputSchema map[string]any   `json:"inputSchema"`
    Handler     ToolHandler      `json:"-"`
}

type ToolHandler func(ctx context.Context, args map[string]any) (any, error)

func NewServer(engine *inference.Engine) *Server {
    s := &Server{
        engine:  engine,
        tools:   make(map[string]*Tool),
        prompts: make(map[string]*Prompt),
    }
    s.registerDefaultTools()
    return s
}

func (s *Server) registerDefaultTools() {
    // Classify text
    s.RegisterTool(&Tool{
        Name:        "classify_text",
        Description: "Classify text into predefined categories using the loaded model",
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "text": map[string]any{
                    "type":        "string",
                    "description": "Text to classify",
                },
                "model_id": map[string]any{
                    "type":        "string",
                    "description": "Model ID to use for classification",
                },
            },
            "required": []string{"text"},
        },
        Handler: s.handleClassify,
    })

    // Semantic search
    s.RegisterTool(&Tool{
        Name:        "semantic_search",
        Description: "Search documents using semantic similarity",
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "query": map[string]any{
                    "type":        "string",
                    "description": "Search query",
                },
                "top_k": map[string]any{
                    "type":        "integer",
                    "description": "Number of results to return",
                    "default":     5,
                },
            },
            "required": []string{"query"},
        },
        Handler: s.handleSearch,
    })

    // Question answering
    s.RegisterTool(&Tool{
        Name:        "answer_question",
        Description: "Answer a question based on the knowledge in the loaded model",
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "question": map[string]any{
                    "type":        "string",
                    "description": "Question to answer",
                },
                "context": map[string]any{
                    "type":        "string",
                    "description": "Optional context to use for answering",
                },
            },
            "required": []string{"question"},
        },
        Handler: s.handleQA,
    })

    // Analyze document
    s.RegisterTool(&Tool{
        Name:        "analyze_document",
        Description: "Analyze a document and extract key information",
        InputSchema: map[string]any{
            "type": "object",
            "properties": map[string]any{
                "file_path": map[string]any{
                    "type":        "string",
                    "description": "Path to the document",
                },
                "analysis_type": map[string]any{
                    "type":        "string",
                    "enum":        []string{"summary", "entities", "classification", "full"},
                    "description": "Type of analysis to perform",
                },
            },
            "required": []string{"file_path"},
        },
        Handler: s.handleAnalyze,
    })
}

func (s *Server) handleClassify(ctx context.Context, args map[string]any) (any, error) {
    text := args["text"].(string)
    modelID := args["model_id"].(string)

    resp, err := s.engine.Predict(ctx, &inference.PredictRequest{
        ModelID: modelID,
        Input:   map[string]any{"text": text},
    })
    if err != nil {
        return nil, err
    }

    return resp.Result, nil
}

// MCP Protocol handlers
func (s *Server) SetupRoutes(app *fiber.App) {
    // MCP over WebSocket
    app.Get("/mcp", websocket.New(s.handleMCPConnection))

    // MCP over HTTP (for simpler integrations)
    mcp := app.Group("/mcp/v1")
    mcp.Get("/tools", s.listTools)
    mcp.Post("/tools/:name/call", s.callTool)
    mcp.Get("/resources", s.listResources)
    mcp.Get("/resources/:uri", s.getResource)
    mcp.Get("/prompts", s.listPrompts)
    mcp.Get("/prompts/:name", s.getPrompt)
}

func (s *Server) handleMCPConnection(c *websocket.Conn) {
    for {
        messageType, msg, err := c.ReadMessage()
        if err != nil {
            break
        }

        if messageType == websocket.TextMessage {
            var request MCPRequest
            if err := sonic.Unmarshal(msg, &request); err != nil {
                s.sendError(c, err)
                continue
            }

            response := s.handleRequest(&request)
            data, _ := sonic.Marshal(response)
            c.WriteMessage(websocket.TextMessage, data)
        }
    }
}

func (s *Server) handleRequest(req *MCPRequest) *MCPResponse {
    switch req.Method {
    case "tools/list":
        return s.toolsList()
    case "tools/call":
        return s.toolsCall(req.Params)
    case "resources/list":
        return s.resourcesList()
    case "resources/read":
        return s.resourcesRead(req.Params)
    case "prompts/list":
        return s.promptsList()
    case "prompts/get":
        return s.promptsGet(req.Params)
    default:
        return &MCPResponse{Error: &MCPError{Code: -32601, Message: "Method not found"}}
    }
}
```

### 6. Autonomous Agent

```go
// internal/agent/agent.go
package agent

import (
    "context"
    "github.com/panjf2000/ants/v2"
)

type Agent struct {
    engine    *inference.Engine
    mcp       *mcp.Server
    planner   *Planner
    executor  *Executor
    memory    *Memory
    pool      *ants.Pool
}

type Task struct {
    ID          string            `json:"id"`
    Description string            `json:"description"`
    Status      TaskStatus        `json:"status"`
    Steps       []*Step           `json:"steps"`
    Context     map[string]any    `json:"context"`
    Result      any               `json:"result"`
    Error       string            `json:"error,omitempty"`
}

type Step struct {
    ID       string         `json:"id"`
    Action   string         `json:"action"`
    Args     map[string]any `json:"args"`
    Status   StepStatus     `json:"status"`
    Result   any            `json:"result"`
    Duration time.Duration  `json:"duration"`
}

func NewAgent(engine *inference.Engine, mcp *mcp.Server, config *Config) (*Agent, error) {
    pool, err := ants.NewPool(config.MaxConcurrency)
    if err != nil {
        return nil, err
    }

    return &Agent{
        engine:   engine,
        mcp:      mcp,
        planner:  NewPlanner(engine),
        executor: NewExecutor(mcp),
        memory:   NewMemory(config.MemorySize),
        pool:     pool,
    }, nil
}

func (a *Agent) ExecuteTask(ctx context.Context, description string) (*Task, error) {
    task := &Task{
        ID:          uuid.New().String(),
        Description: description,
        Status:      TaskStatusPlanning,
        Context:     make(map[string]any),
    }

    // Plan the task
    steps, err := a.planner.Plan(ctx, description, a.memory.GetRelevant(description))
    if err != nil {
        task.Status = TaskStatusFailed
        task.Error = err.Error()
        return task, err
    }
    task.Steps = steps
    task.Status = TaskStatusRunning

    // Execute steps
    for _, step := range task.Steps {
        step.Status = StepStatusRunning
        start := time.Now()

        result, err := a.executor.Execute(ctx, step)
        step.Duration = time.Since(start)

        if err != nil {
            step.Status = StepStatusFailed
            task.Status = TaskStatusFailed
            task.Error = err.Error()
            return task, err
        }

        step.Result = result
        step.Status = StepStatusCompleted

        // Update context for next step
        task.Context[step.ID] = result

        // Store in memory
        a.memory.Store(step)
    }

    task.Status = TaskStatusCompleted
    task.Result = task.Context

    return task, nil
}
```

### 7. Filesystem Collector

```go
// internal/collector/scanner.go
package collector

import (
    "io/fs"
    "path/filepath"
    "github.com/h2non/filetype"
)

type Scanner struct {
    config  *Config
    filters *Filters
}

type FileInfo struct {
    Path      string            `json:"path"`
    Name      string            `json:"name"`
    Size      int64             `json:"size"`
    ModTime   time.Time         `json:"modTime"`
    MimeType  string            `json:"mimeType"`
    Extension string            `json:"extension"`
    Hash      string            `json:"hash,omitempty"`
    Metadata  map[string]string `json:"metadata,omitempty"`
}

type ScanResult struct {
    Files      []FileInfo `json:"files"`
    TotalSize  int64      `json:"totalSize"`
    TotalFiles int        `json:"totalFiles"`
    Duration   string     `json:"duration"`
    Errors     []string   `json:"errors,omitempty"`
}

func (s *Scanner) Scan(ctx context.Context, paths []string) (*ScanResult, error) {
    start := time.Now()
    result := &ScanResult{}

    var mu sync.Mutex
    var wg sync.WaitGroup
    sem := make(chan struct{}, s.config.MaxConcurrency)

    for _, root := range paths {
        err := filepath.WalkDir(root, func(path string, d fs.DirEntry, err error) error {
            select {
            case <-ctx.Done():
                return ctx.Err()
            default:
            }

            if err != nil {
                mu.Lock()
                result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", path, err))
                mu.Unlock()
                return nil
            }

            if d.IsDir() {
                // Check skip patterns
                if s.filters.ShouldSkipDir(d.Name()) {
                    return filepath.SkipDir
                }
                return nil
            }

            // Apply file filters
            if !s.filters.Matches(path, d) {
                return nil
            }

            wg.Add(1)
            sem <- struct{}{}

            go func(p string, entry fs.DirEntry) {
                defer wg.Done()
                defer func() { <-sem }()

                info, err := s.processFile(p, entry)
                if err != nil {
                    mu.Lock()
                    result.Errors = append(result.Errors, fmt.Sprintf("%s: %v", p, err))
                    mu.Unlock()
                    return
                }

                mu.Lock()
                result.Files = append(result.Files, info)
                result.TotalSize += info.Size
                result.TotalFiles++
                mu.Unlock()
            }(path, d)

            return nil
        })

        if err != nil {
            return nil, err
        }
    }

    wg.Wait()
    result.Duration = time.Since(start).String()

    return result, nil
}

func (s *Scanner) processFile(path string, entry fs.DirEntry) (FileInfo, error) {
    stat, err := entry.Info()
    if err != nil {
        return FileInfo{}, err
    }

    // Detect MIME type
    mimeType := s.detectMimeType(path)

    info := FileInfo{
        Path:      path,
        Name:      entry.Name(),
        Size:      stat.Size(),
        ModTime:   stat.ModTime(),
        MimeType:  mimeType,
        Extension: filepath.Ext(path),
    }

    // Calculate hash if enabled
    if s.config.CalculateHash {
        hash, err := s.calculateHash(path)
        if err == nil {
            info.Hash = hash
        }
    }

    return info, nil
}

func (s *Scanner) detectMimeType(path string) string {
    file, err := os.Open(path)
    if err != nil {
        return "application/octet-stream"
    }
    defer file.Close()

    // Read first 512 bytes for detection
    head := make([]byte, 512)
    n, _ := file.Read(head)

    kind, err := filetype.Match(head[:n])
    if err != nil || kind == filetype.Unknown {
        return "application/octet-stream"
    }

    return kind.MIME.Value
}
```

### 8. Storage Layer (BadgerDB)

```go
// internal/storage/db.go
package storage

import (
    "github.com/dgraph-io/badger/v4"
    "github.com/bytedance/sonic"
)

type DB struct {
    db *badger.DB
}

func NewDB(path string) (*DB, error) {
    opts := badger.DefaultOptions(path).
        WithLoggingLevel(badger.WARNING).
        WithCompression(options.ZSTD).
        WithBlockCacheSize(256 << 20). // 256MB block cache
        WithIndexCacheSize(128 << 20)  // 128MB index cache

    db, err := badger.Open(opts)
    if err != nil {
        return nil, err
    }

    // Start GC routine
    go func() {
        ticker := time.NewTicker(5 * time.Minute)
        defer ticker.Stop()
        for range ticker.C {
            db.RunValueLogGC(0.5)
        }
    }()

    return &DB{db: db}, nil
}

func (d *DB) Set(key string, value any) error {
    data, err := sonic.Marshal(value)
    if err != nil {
        return err
    }

    return d.db.Update(func(txn *badger.Txn) error {
        return txn.Set([]byte(key), data)
    })
}

func (d *DB) Get(key string, dest any) error {
    return d.db.View(func(txn *badger.Txn) error {
        item, err := txn.Get([]byte(key))
        if err != nil {
            return err
        }
        return item.Value(func(val []byte) error {
            return sonic.Unmarshal(val, dest)
        })
    })
}
```

## Build & Distribution

### Cross-Compilation

```bash
#!/bin/bash
# scripts/build.sh

VERSION=${1:-"dev"}
LDFLAGS="-s -w -X main.Version=$VERSION"

# Build for all platforms
GOOS=darwin GOARCH=amd64 go build -ldflags="$LDFLAGS" -o dist/ads-agent-darwin-amd64 ./cmd/agent
GOOS=darwin GOARCH=arm64 go build -ldflags="$LDFLAGS" -o dist/ads-agent-darwin-arm64 ./cmd/agent
GOOS=linux GOARCH=amd64 go build -ldflags="$LDFLAGS" -o dist/ads-agent-linux-amd64 ./cmd/agent
GOOS=linux GOARCH=arm64 go build -ldflags="$LDFLAGS" -o dist/ads-agent-linux-arm64 ./cmd/agent
GOOS=windows GOARCH=amd64 go build -ldflags="$LDFLAGS" -o dist/ads-agent-windows-amd64.exe ./cmd/agent

# Compress with UPX (optional)
upx --best dist/ads-agent-*
```

### Configuration File

```yaml
# configs/agent.yaml
server:
  host: "127.0.0.1"
  port: 8484

inference:
  max_concurrent: 4
  cache_size_mb: 1024
  cache_ttl_minutes: 10
  enable_gpu: false
  onnx_lib_path: ""

models:
  storage_path: "~/.ads-agent/models"
  auto_load: []

mcp:
  enabled: true
  websocket_enabled: true

agent:
  enabled: true
  max_concurrency: 2
  memory_size: 1000

backend:
  url: "https://api.aiserve.farm"
  api_key: ""

collector:
  max_file_size_mb: 100
  calculate_hash: true
  max_concurrency: 8
  skip_patterns:
    - ".git"
    - "node_modules"
    - "__pycache__"
    - ".venv"

logging:
  level: "info"
  format: "json"
```

## Performance Benchmarks (Targets)

```
BenchmarkHealthCheck-8          1000000    980 ns/op       0 B/op    0 allocs/op
BenchmarkJSONMarshal-8           500000   2100 ns/op     128 B/op    2 allocs/op
BenchmarkCacheGet-8             2000000    650 ns/op       0 B/op    0 allocs/op
BenchmarkONNXInference-8          10000 125000 ns/op    4096 B/op   12 allocs/op
BenchmarkFileScan1000-8              20 45000000 ns/op  512000 B/op 1200 allocs/op
```
