# ADS Agent - Go Binary
# Multi-stage build for minimal image

# Build stage
FROM golang:1.22-alpine AS builder

WORKDIR /app

# Install build dependencies
RUN apk add --no-cache git ca-certificates

# Copy go mod files
COPY go.mod go.sum ./
RUN go mod download

# Copy source
COPY . .

# Build binary
RUN CGO_ENABLED=0 GOOS=linux go build \
    -ldflags="-s -w -X main.Version=$(git describe --tags --always 2>/dev/null || echo dev)" \
    -o /ads-agent \
    ./cmd/agent

# Runtime stage
FROM alpine:3.19

WORKDIR /app

# Install runtime dependencies
RUN apk add --no-cache ca-certificates tzdata

# Copy binary
COPY --from=builder /ads-agent /usr/local/bin/ads-agent

# Create non-root user
RUN adduser -D -u 1000 appuser
USER appuser

# Create data directory
RUN mkdir -p /home/appuser/.ads-agent

# Expose port
EXPOSE 8484

# Health check
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
    CMD wget -qO- http://localhost:8484/health || exit 1

# Run
ENTRYPOINT ["ads-agent"]
