// ADS Agent - Local model serving and data collection
//
// High-performance inference engine with MCP support.
// Designed for zero-dependency deployment.
package main

import (
	"context"
	"flag"
	"os"
	"os/signal"
	"syscall"

	"github.com/rs/zerolog"
	"github.com/rs/zerolog/log"

	"github.com/afterdarksolutions/ads-agent/internal/api"
	"github.com/afterdarksolutions/ads-agent/internal/config"
	"github.com/afterdarksolutions/ads-agent/internal/inference"
	"github.com/afterdarksolutions/ads-agent/internal/storage"
)

var (
	Version   = "dev"
	BuildTime = "unknown"
)

func main() {
	// Flags
	configPath := flag.String("config", "", "Path to config file")
	verbose := flag.Bool("verbose", false, "Enable verbose logging")
	version := flag.Bool("version", false, "Print version and exit")
	flag.Parse()

	if *version {
		println("ADS Agent", Version, "built", BuildTime)
		os.Exit(0)
	}

	// Setup logging
	setupLogging(*verbose)

	log.Info().
		Str("version", Version).
		Msg("Starting ADS Agent")

	// Load config
	cfg, err := config.Load(*configPath)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to load config")
	}

	// Initialize components
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	// Storage (BadgerDB for local persistence)
	store, err := storage.NewDB(cfg.Storage.Path)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to initialize storage")
	}
	defer store.Close()

	// Inference engine
	engine, err := inference.NewEngine(cfg.Inference)
	if err != nil {
		log.Fatal().Err(err).Msg("Failed to initialize inference engine")
	}
	defer engine.Close()

	// Auto-load models
	for _, modelPath := range cfg.Models.AutoLoad {
		if err := engine.LoadModel(ctx, modelPath); err != nil {
			log.Warn().Err(err).Str("path", modelPath).Msg("Failed to auto-load model")
		}
	}

	// HTTP server
	server := api.NewServer(cfg.Server, engine, store, cfg.Models.StoragePath)
	server.SetupRoutes()

	// Graceful shutdown
	go func() {
		sigCh := make(chan os.Signal, 1)
		signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
		<-sigCh

		log.Info().Msg("Shutting down...")
		cancel()

		if err := server.Shutdown(); err != nil {
			log.Error().Err(err).Msg("Server shutdown error")
		}
	}()

	// Start server
	if err := server.Start(); err != nil {
		log.Fatal().Err(err).Msg("Server error")
	}
}

func setupLogging(verbose bool) {
	// Pretty console output
	output := zerolog.ConsoleWriter{Out: os.Stderr}
	log.Logger = zerolog.New(output).With().Timestamp().Logger()

	if verbose {
		zerolog.SetGlobalLevel(zerolog.DebugLevel)
	} else {
		zerolog.SetGlobalLevel(zerolog.InfoLevel)
	}
}
