// Package config handles agent configuration.
package config

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"github.com/spf13/viper"
)

// Config is the main configuration structure.
type Config struct {
	Server    ServerConfig    `mapstructure:"server"`
	Inference InferenceConfig `mapstructure:"inference"`
	Models    ModelsConfig    `mapstructure:"models"`
	Storage   StorageConfig   `mapstructure:"storage"`
	MCP       MCPConfig       `mapstructure:"mcp"`
	Agent     AgentConfig     `mapstructure:"agent"`
	Backend   BackendConfig   `mapstructure:"backend"`
}

// ServerConfig for the HTTP API.
type ServerConfig struct {
	Host         string `mapstructure:"host"`
	Port         int    `mapstructure:"port"`
	ReadTimeout  int    `mapstructure:"read_timeout"`
	WriteTimeout int    `mapstructure:"write_timeout"`
	Prefork      bool   `mapstructure:"prefork"`
	APIKey       string `mapstructure:"api_key"`
	MaxBodyBytes int    `mapstructure:"max_body_bytes"`
}

// InferenceConfig for the inference engine.
type InferenceConfig struct {
	MaxConcurrent int    `mapstructure:"max_concurrent"`
	CacheSizeMB   int    `mapstructure:"cache_size_mb"`
	CacheTTLMin   int    `mapstructure:"cache_ttl_minutes"`
	EnableGPU     bool   `mapstructure:"enable_gpu"`
	ONNXLibPath   string `mapstructure:"onnx_lib_path"`
}

// ModelsConfig for model management.
type ModelsConfig struct {
	StoragePath string   `mapstructure:"storage_path"`
	AutoLoad    []string `mapstructure:"auto_load"`
}

// StorageConfig for local persistence.
type StorageConfig struct {
	Path string `mapstructure:"path"`
}

// MCPConfig for Model Context Protocol support.
type MCPConfig struct {
	Enabled          bool `mapstructure:"enabled"`
	WebSocketEnabled bool `mapstructure:"websocket_enabled"`
}

// AgentConfig for autonomous agent capabilities.
type AgentConfig struct {
	Enabled        bool `mapstructure:"enabled"`
	MaxConcurrency int  `mapstructure:"max_concurrency"`
	MemorySize     int  `mapstructure:"memory_size"`
}

// BackendConfig for cloud backend connection.
type BackendConfig struct {
	URL    string `mapstructure:"url"`
	APIKey string `mapstructure:"api_key"`
}

// Load configuration from file and environment.
func Load(configPath string) (*Config, error) {
	v := viper.New()

	// Set defaults
	setDefaults(v)

	// Config file
	if configPath != "" {
		v.SetConfigFile(configPath)
	} else {
		// Look in standard locations
		v.SetConfigName("agent")
		v.SetConfigType("yaml")
		v.AddConfigPath(".")
		v.AddConfigPath(filepath.Join(homeDir(), ".ads-agent"))
		v.AddConfigPath("/etc/ads-agent")
	}

	// Environment variables. Nested keys use ADS_SERVER_HOST, not ADS_SERVER.HOST.
	v.SetEnvPrefix("ADS")
	v.SetEnvKeyReplacer(strings.NewReplacer(".", "_"))
	v.AutomaticEnv()

	// Read config
	if err := v.ReadInConfig(); err != nil {
		if _, ok := err.(viper.ConfigFileNotFoundError); !ok {
			return nil, err
		}
		// Config file not found, use defaults
	}

	var cfg Config
	if err := v.Unmarshal(&cfg); err != nil {
		return nil, err
	}

	if err := applyEnvOverrides(&cfg); err != nil {
		return nil, err
	}

	// Expand paths
	cfg.Models.StoragePath = expandPath(cfg.Models.StoragePath)
	cfg.Storage.Path = expandPath(cfg.Storage.Path)

	return &cfg, nil
}

// applyEnvOverrides copies the variables the process is actually given.
// Viper's AutomaticEnv does not reliably fill Unmarshal for nested keys.
func applyEnvOverrides(cfg *Config) error {
	if value := os.Getenv("ADS_SERVER_HOST"); value != "" {
		cfg.Server.Host = value
	}
	if value := os.Getenv("ADS_SERVER_PORT"); value != "" {
		port, err := strconv.Atoi(value)
		if err != nil || port < 1 || port > 65535 {
			return fmt.Errorf("ADS_SERVER_PORT is invalid")
		}
		cfg.Server.Port = port
	}
	if value := os.Getenv("ADS_SERVER_API_KEY"); value != "" {
		cfg.Server.APIKey = value
	}
	if value := os.Getenv("ADS_SERVER_MAX_BODY_BYTES"); value != "" {
		size, err := strconv.Atoi(value)
		if err != nil || size < 1 {
			return fmt.Errorf("ADS_SERVER_MAX_BODY_BYTES is invalid")
		}
		cfg.Server.MaxBodyBytes = size
	}
	if value := os.Getenv("ADS_LOG_LEVEL"); value != "" {
		// Accepted so a set variable is not silently ignored. Logging is
		// configured by the process before config load.
		_ = value
	}
	return nil
}

func setDefaults(v *viper.Viper) {
	// Server
	v.SetDefault("server.host", "127.0.0.1")
	v.SetDefault("server.port", 8484)
	v.SetDefault("server.read_timeout", 5)
	v.SetDefault("server.write_timeout", 10)
	v.SetDefault("server.prefork", false)
	v.SetDefault("server.api_key", "")
	v.SetDefault("server.max_body_bytes", 1<<20)

	// Inference
	v.SetDefault("inference.max_concurrent", 4)
	v.SetDefault("inference.cache_size_mb", 512)
	v.SetDefault("inference.cache_ttl_minutes", 10)
	v.SetDefault("inference.enable_gpu", false)
	v.SetDefault("inference.onnx_lib_path", "")

	// Models
	v.SetDefault("models.storage_path", "~/.ads-agent/models")
	v.SetDefault("models.auto_load", []string{})

	// Storage
	v.SetDefault("storage.path", "~/.ads-agent/data")

	// MCP
	v.SetDefault("mcp.enabled", true)
	v.SetDefault("mcp.websocket_enabled", true)

	// Agent
	v.SetDefault("agent.enabled", true)
	v.SetDefault("agent.max_concurrency", 2)
	v.SetDefault("agent.memory_size", 1000)

	// Backend
	v.SetDefault("backend.url", "https://api.aiserve.farm")
	v.SetDefault("backend.api_key", "")
}

func homeDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		return "."
	}
	return home
}

func expandPath(path string) string {
	if len(path) == 0 {
		return path
	}
	if path[0] == '~' {
		return filepath.Join(homeDir(), path[1:])
	}
	return path
}
