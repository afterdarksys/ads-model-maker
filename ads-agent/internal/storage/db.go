// Package storage provides local persistence using BadgerDB.
package storage

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"

	"github.com/dgraph-io/badger/v4"
	"github.com/rs/zerolog/log"
)

// DB wraps BadgerDB for local storage.
type DB struct {
	db        *badger.DB
	path      string
	stop      chan struct{}
	closeOnce sync.Once
}

// NewDB opens or creates a BadgerDB database.
func NewDB(path string) (*DB, error) {
	// Ensure directory exists
	if err := os.MkdirAll(path, 0755); err != nil {
		return nil, fmt.Errorf("failed to create storage directory: %w", err)
	}

	opts := badger.DefaultOptions(path).
		WithLoggingLevel(badger.WARNING).
		WithNumVersionsToKeep(1).
		WithCompactL0OnClose(true).
		WithValueLogFileSize(64 << 20). // 64MB
		WithBlockCacheSize(64 << 20).   // 64MB
		WithIndexCacheSize(32 << 20)    // 32MB

	db, err := badger.Open(opts)
	if err != nil {
		return nil, fmt.Errorf("failed to open database: %w", err)
	}

	store := &DB{
		db:   db,
		path: path,
		stop: make(chan struct{}),
	}

	// Start GC routine
	go store.runGC()

	log.Info().Str("path", path).Msg("Storage initialized")
	return store, nil
}

// Close closes the database and stops the value-log garbage collector.
func (d *DB) Close() error {
	var err error
	d.closeOnce.Do(func() {
		close(d.stop)
		err = d.db.Close()
	})
	return err
}

// Set stores a value.
func (d *DB) Set(key string, value interface{}) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}

	return d.db.Update(func(txn *badger.Txn) error {
		return txn.Set([]byte(key), data)
	})
}

// SetWithTTL stores a value with expiration.
func (d *DB) SetWithTTL(key string, value interface{}, ttl time.Duration) error {
	data, err := json.Marshal(value)
	if err != nil {
		return fmt.Errorf("failed to marshal value: %w", err)
	}

	return d.db.Update(func(txn *badger.Txn) error {
		entry := badger.NewEntry([]byte(key), data).WithTTL(ttl)
		return txn.SetEntry(entry)
	})
}

// Get retrieves a value.
func (d *DB) Get(key string, dest interface{}) error {
	return d.db.View(func(txn *badger.Txn) error {
		item, err := txn.Get([]byte(key))
		if err != nil {
			return err
		}

		return item.Value(func(val []byte) error {
			return json.Unmarshal(val, dest)
		})
	})
}

// Delete removes a key.
func (d *DB) Delete(key string) error {
	return d.db.Update(func(txn *badger.Txn) error {
		return txn.Delete([]byte(key))
	})
}

// Exists checks if a key exists.
func (d *DB) Exists(key string) bool {
	err := d.db.View(func(txn *badger.Txn) error {
		_, err := txn.Get([]byte(key))
		return err
	})
	return err == nil
}

// Keys returns all keys with a prefix.
func (d *DB) Keys(prefix string) ([]string, error) {
	var keys []string

	err := d.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		opts.PrefetchValues = false

		it := txn.NewIterator(opts)
		defer it.Close()

		prefixBytes := []byte(prefix)
		for it.Seek(prefixBytes); it.ValidForPrefix(prefixBytes); it.Next() {
			item := it.Item()
			keys = append(keys, string(item.Key()))
		}

		return nil
	})

	return keys, err
}

// Scan iterates over keys with a prefix.
func (d *DB) Scan(prefix string, fn func(key string, value []byte) error) error {
	return d.db.View(func(txn *badger.Txn) error {
		opts := badger.DefaultIteratorOptions
		it := txn.NewIterator(opts)
		defer it.Close()

		prefixBytes := []byte(prefix)
		for it.Seek(prefixBytes); it.ValidForPrefix(prefixBytes); it.Next() {
			item := it.Item()
			key := string(item.Key())

			err := item.Value(func(val []byte) error {
				return fn(key, val)
			})
			if err != nil {
				return err
			}
		}

		return nil
	})
}

// Stats returns database statistics.
func (d *DB) Stats() map[string]interface{} {
	lsm, vlog := d.db.Size()

	return map[string]interface{}{
		"lsm_size_bytes":  lsm,
		"vlog_size_bytes": vlog,
		"total_size_mb":   float64(lsm+vlog) / (1024 * 1024),
	}
}

func (d *DB) runGC() {
	ticker := time.NewTicker(5 * time.Minute)
	defer ticker.Stop()

	for {
		select {
		case <-d.stop:
			return
		case <-ticker.C:
			err := d.db.RunValueLogGC(0.5)
			if err != nil && err != badger.ErrNoRewrite {
				log.Debug().Err(err).Msg("BadgerDB GC error")
			}
		}
	}
}

// ============================================================================
// Model Storage
// ============================================================================

// ModelMeta stores model metadata.
type ModelMeta struct {
	ID        string    `json:"id"`
	Name      string    `json:"name"`
	Type      string    `json:"type"`
	Path      string    `json:"path"`
	Size      int64     `json:"size"`
	CreatedAt time.Time `json:"created_at"`
	Config    string    `json:"config,omitempty"`
}

// SaveModel saves model metadata.
func (d *DB) SaveModel(model *ModelMeta) error {
	key := fmt.Sprintf("model:%s", model.ID)
	return d.Set(key, model)
}

// GetModel retrieves model metadata.
func (d *DB) GetModel(id string) (*ModelMeta, error) {
	key := fmt.Sprintf("model:%s", id)
	var model ModelMeta
	if err := d.Get(key, &model); err != nil {
		return nil, err
	}
	return &model, nil
}

// ListModels returns all stored models.
func (d *DB) ListModels() ([]*ModelMeta, error) {
	var models []*ModelMeta

	err := d.Scan("model:", func(key string, value []byte) error {
		var model ModelMeta
		if err := json.Unmarshal(value, &model); err != nil {
			return err
		}
		models = append(models, &model)
		return nil
	})

	return models, err
}

// DeleteModel removes model metadata.
func (d *DB) DeleteModel(id string) error {
	key := fmt.Sprintf("model:%s", id)
	return d.Delete(key)
}

// ============================================================================
// Cache Storage
// ============================================================================

// CacheEntry for inference results.
type CacheEntry struct {
	Key       string      `json:"key"`
	Value     interface{} `json:"value"`
	CreatedAt time.Time   `json:"created_at"`
}

// SetCache stores a cache entry.
func (d *DB) SetCache(key string, value interface{}, ttl time.Duration) error {
	entry := CacheEntry{
		Key:       key,
		Value:     value,
		CreatedAt: time.Now(),
	}
	cacheKey := fmt.Sprintf("cache:%s", key)
	return d.SetWithTTL(cacheKey, entry, ttl)
}

// GetCache retrieves a cache entry.
func (d *DB) GetCache(key string) (interface{}, error) {
	cacheKey := fmt.Sprintf("cache:%s", key)
	var entry CacheEntry
	if err := d.Get(cacheKey, &entry); err != nil {
		return nil, err
	}
	return entry.Value, nil
}

// ClearCache removes all cache entries.
func (d *DB) ClearCache() error {
	keys, err := d.Keys("cache:")
	if err != nil {
		return err
	}

	for _, key := range keys {
		if err := d.Delete(key); err != nil {
			return err
		}
	}

	return nil
}

// GetStorageDir returns the storage directory path.
func (d *DB) GetStorageDir() string {
	return d.path
}

// GetModelsDir returns the models subdirectory.
func (d *DB) GetModelsDir() string {
	return filepath.Join(d.path, "models")
}
