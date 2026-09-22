package config

import "testing"

func TestInvalidPortFromEnvIsRejected(t *testing.T) {
	t.Setenv("ADS_SERVER_PORT", "nope")
	if _, err := Load(""); err == nil {
		t.Fatal("invalid ADS_SERVER_PORT was accepted")
	}
}

func TestAPIKeyFromEnvIsApplied(t *testing.T) {
	t.Setenv("ADS_SERVER_PORT", "")
	t.Setenv("ADS_SERVER_API_KEY", "local-test-key")
	cfg, err := Load("")
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Server.APIKey != "local-test-key" {
		t.Fatalf("api key = %q", cfg.Server.APIKey)
	}
}
