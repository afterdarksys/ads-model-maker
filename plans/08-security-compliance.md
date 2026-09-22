# Security & Compliance

## Security Philosophy

1. **Defense in Depth** - Multiple layers of security
2. **Least Privilege** - Minimal access by default
3. **Zero Trust** - Verify everything, trust nothing
4. **Privacy by Design** - Data protection built-in

## Data Security

### Data Classification

| Level | Description | Examples | Handling |
|-------|-------------|----------|----------|
| Public | Non-sensitive | Marketing content | Standard encryption |
| Internal | Business data | Training metrics | Encrypted at rest |
| Confidential | Customer data | Uploaded documents | E2E encryption |
| Restricted | Regulated data | PHI, PII, financial | E2E + access controls |

### Encryption Standards

#### At Rest
- **Storage**: AES-256-GCM
- **Database**: Transparent Data Encryption (TDE)
- **Backups**: Encrypted with separate keys
- **Key Management**: HashiCorp Vault / AWS KMS

#### In Transit
- **API**: TLS 1.3 (minimum TLS 1.2)
- **Internal**: mTLS between services
- **Agent ↔ Backend**: Certificate pinning
- **WebSocket**: WSS only

#### End-to-End (Compliance Tiers)
```
User Data → Agent Encryption → Transit → Storage (encrypted)
                    ↓
            Client-side keys
            (never leave user device for E2E tiers)
```

### Data Lifecycle

```
┌─────────────┐
│   Upload    │──► Encrypted in transit
└──────┬──────┘
       │
       ▼
┌─────────────┐
│  Ingest     │──► Processed in isolated container
└──────┬──────┘    ──► Memory cleared after processing
       │
       ▼
┌─────────────┐
│  Storage    │──► Encrypted at rest
└──────┬──────┘    ──► Access logged
       │
       ▼
┌─────────────┐
│  Training   │──► Isolated GPU environment
└──────┬──────┘    ──► Data deleted after training (optional)
       │
       ▼
┌─────────────┐
│   Model     │──► Model stored encrypted
└──────┬──────┘    ──► User controls access
       │
       ▼
┌─────────────┐
│  Deletion   │──► Cryptographic erasure
└─────────────┘    ──► Audit trail preserved
```

### Data Isolation

- **Tenant Isolation**: Separate encryption keys per tenant
- **Process Isolation**: Containers for training jobs
- **Network Isolation**: VPC segmentation
- **Storage Isolation**: Separate buckets/prefixes per user

## Authentication & Authorization

### Authentication Methods

#### API Keys
```
Format: ads_sk_[base58(32 bytes)]
Example: ads_sk_4xK9mNvP3qR7sT2wY6zB8cD1fG5hJ0kL

Properties:
- Revocable
- Scoped permissions
- Rate limited
- Audit logged
```

#### JWT Tokens
```json
{
  "sub": "user_abc123",
  "iat": 1705764000,
  "exp": 1705767600,
  "scope": ["read:models", "write:training"],
  "tier": "professional"
}
```

- Short expiry (1 hour)
- Refresh token rotation
- Secure httpOnly cookies for web

#### SSO (Enterprise Tiers)
- SAML 2.0
- OIDC
- Azure AD integration
- Okta integration

### Authorization Model

```yaml
# RBAC Permissions
roles:
  viewer:
    - read:models
    - read:data_sources
    - read:jobs

  developer:
    - include: viewer
    - write:training
    - write:inference
    - delete:own_models

  admin:
    - include: developer
    - manage:team
    - manage:billing
    - delete:any

  owner:
    - include: admin
    - manage:organization
    - transfer:ownership
```

### Multi-Factor Authentication

Required for:
- Compliance tier accounts (Legal, Medical, Gov, Finance)
- Sensitive operations (delete all, export data)
- Admin actions

Supported methods:
- TOTP (Google Authenticator, Authy)
- SMS (backup only)
- Hardware keys (FIDO2/WebAuthn)

## Infrastructure Security

### Network Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        PUBLIC INTERNET                       │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                     WAF / DDoS Protection                    │
│                    (Cloudflare / AWS Shield)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                       Load Balancer                          │
│                     (Public Subnet)                          │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                      API Gateway                             │
│                    (Private Subnet)                          │
└─────────────────────────────────────────────────────────────┘
                              │
         ┌────────────────────┼────────────────────┐
         ▼                    ▼                    ▼
┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│  Application    │  │   Training      │  │    Database     │
│   Services      │  │   Workers       │  │    Cluster      │
│ (Private Subnet)│  │(Isolated Subnet)│  │(Private Subnet) │
└─────────────────┘  └─────────────────┘  └─────────────────┘
```

### Security Groups

```hcl
# API servers - minimal exposure
resource "aws_security_group" "api" {
  ingress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["10.0.0.0/16"]  # Internal only
  }
}

# Database - no public access
resource "aws_security_group" "database" {
  ingress {
    from_port       = 5432
    to_port         = 5432
    protocol        = "tcp"
    security_groups = [aws_security_group.api.id]
  }
}

# Training workers - isolated
resource "aws_security_group" "training" {
  # Egress only to storage and HF
  egress {
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
}
```

### Container Security

```dockerfile
# Non-root user
RUN adduser -D -u 1000 appuser
USER appuser

# Read-only filesystem
# (configured at runtime)

# No new privileges
# securityContext:
#   allowPrivilegeEscalation: false
#   readOnlyRootFilesystem: true
#   runAsNonRoot: true
```

### Secrets Management

- **HashiCorp Vault** for production secrets
- **Environment injection** at runtime
- **Rotation policy**: 90 days for API keys
- **No secrets in code or config files**

```go
// Example: Vault integration
func getSecret(path string) (string, error) {
    client, err := vault.NewClient(vault.DefaultConfig())
    if err != nil {
        return "", err
    }

    secret, err := client.Logical().Read(path)
    if err != nil {
        return "", err
    }

    return secret.Data["value"].(string), nil
}
```

## Application Security

### Input Validation

```python
from pydantic import BaseModel, validator, constr

class TrainingRequest(BaseModel):
    model_type: Literal["classification", "qa", "embedding"]
    data_source_ids: list[UUID]
    config: TrainingConfig

    @validator('data_source_ids')
    def validate_sources(cls, v):
        if len(v) > 10:
            raise ValueError('Maximum 10 data sources per job')
        return v

    @validator('config')
    def validate_config(cls, v):
        if v.epochs > 100:
            raise ValueError('Maximum 100 epochs')
        return v
```

### Output Encoding

```python
# Always escape user-generated content
from markupsafe import escape

def render_model_name(name: str) -> str:
    return escape(name)
```

### SQL Injection Prevention

```python
# Always use parameterized queries
async def get_user_models(user_id: UUID) -> list[Model]:
    query = """
        SELECT * FROM models
        WHERE user_id = $1
        ORDER BY created_at DESC
    """
    return await db.fetch_all(query, user_id)  # Parameterized
```

### Rate Limiting

```go
// Fiber rate limiting middleware
app.Use(limiter.New(limiter.Config{
    Max:        100,
    Expiration: 1 * time.Minute,
    KeyGenerator: func(c *fiber.Ctx) string {
        return c.Get("X-API-Key")
    },
    LimitReached: func(c *fiber.Ctx) error {
        return c.Status(429).JSON(fiber.Map{
            "error": "Rate limit exceeded",
        })
    },
}))
```

### CORS Configuration

```go
app.Use(cors.New(cors.Config{
    AllowOrigins:     "https://app.aiserve.farm",
    AllowMethods:     "GET,POST,DELETE",
    AllowHeaders:     "Authorization,Content-Type",
    AllowCredentials: true,
    MaxAge:           3600,
}))
```

## Audit & Logging

### Audit Events

| Event | Logged Data | Retention |
|-------|-------------|-----------|
| User login | User ID, IP, method, success | 2 years |
| Data upload | User ID, file hash, size | 7 years |
| Training start | Job ID, config, user ID | 7 years |
| Model access | User ID, model ID, action | 7 years |
| Data deletion | User ID, resource ID, timestamp | Permanent |
| Permission change | Actor, target, old/new | Permanent |

### Log Format

```json
{
  "timestamp": "2024-01-20T15:30:00.000Z",
  "level": "info",
  "event": "model.accessed",
  "actor": {
    "user_id": "user_abc123",
    "ip": "192.168.1.100",
    "user_agent": "ADS-Agent/1.0"
  },
  "resource": {
    "type": "model",
    "id": "model_xyz789"
  },
  "action": "download",
  "result": "success",
  "metadata": {
    "format": "safetensors",
    "size_bytes": 125000000
  }
}
```

### Log Security

- Logs encrypted at rest
- Separate access controls
- Tamper-evident (append-only)
- No PII in logs (tokenized)

## Compliance Frameworks

### SOC 2 Type II

**Controls implemented:**

| Category | Control | Implementation |
|----------|---------|----------------|
| CC6.1 | Logical access | RBAC, MFA, API keys |
| CC6.2 | Authentication | JWT, SSO, key rotation |
| CC6.3 | Authorization | Permission scoping |
| CC7.1 | System operations | Monitoring, alerting |
| CC7.2 | Change management | Git, code review, CI/CD |
| CC8.1 | Risk assessment | Regular security audits |

### HIPAA (Medical Tier)

**Technical safeguards:**

| Requirement | Implementation |
|-------------|----------------|
| Access control | RBAC + MFA required |
| Audit controls | Comprehensive logging |
| Integrity | Checksums, tamper detection |
| Transmission security | TLS 1.3, E2E encryption |
| Encryption | AES-256 at rest |

**Administrative safeguards:**

- Business Associate Agreement (BAA)
- Workforce training documentation
- Incident response procedures
- Risk analysis records

### FedRAMP (Government Tier)

**Moderate baseline controls:**

- 325 controls from NIST 800-53
- Continuous monitoring
- Third-party assessment (3PAO)
- US-only data residency
- Background checks for staff

### PCI-DSS (Finance Tier)

**Requirements addressed:**

| Requirement | Implementation |
|-------------|----------------|
| Req 1: Firewall | Network segmentation |
| Req 3: Protect data | Encryption at rest |
| Req 4: Encrypt transmission | TLS 1.3 |
| Req 6: Secure systems | Vulnerability management |
| Req 7: Restrict access | RBAC |
| Req 8: Identify users | Unique IDs, MFA |
| Req 10: Track access | Audit logging |
| Req 11: Test security | Penetration testing |
| Req 12: Security policy | Documented policies |

## Incident Response

### Severity Levels

| Level | Definition | Response Time | Example |
|-------|------------|---------------|---------|
| P1 - Critical | Service down, data breach | 15 minutes | Active breach |
| P2 - High | Major feature broken | 1 hour | Training failing |
| P3 - Medium | Degraded performance | 4 hours | Slow inference |
| P4 - Low | Minor issue | 24 hours | UI bug |

### Response Procedures

#### Data Breach Response

```
1. Detect & Identify (0-1 hour)
   - Confirm breach occurred
   - Identify scope and affected data
   - Preserve evidence

2. Contain (1-4 hours)
   - Isolate affected systems
   - Revoke compromised credentials
   - Block attacker access

3. Notify (4-24 hours)
   - Internal stakeholders
   - Affected customers (as required)
   - Regulators (HIPAA: 60 days, GDPR: 72 hours)

4. Remediate (1-7 days)
   - Patch vulnerabilities
   - Reset credentials
   - Restore from clean backups

5. Review (7-30 days)
   - Root cause analysis
   - Update procedures
   - Document lessons learned
```

### Communication Templates

```markdown
# Customer Notification (Data Breach)

Subject: Important Security Notice Regarding Your ADS Model Maker Account

Dear [Customer Name],

We are writing to inform you of a security incident that may have affected
your data on the ADS Model Maker platform.

**What happened:**
[Brief description]

**What data was involved:**
[Specific data types]

**What we are doing:**
[Remediation steps]

**What you can do:**
[Recommended actions]

**For more information:**
[Contact details]

We sincerely apologize for this incident and are committed to protecting
your data.
```

## Security Testing

### Continuous Testing

| Test Type | Frequency | Tools |
|-----------|-----------|-------|
| SAST | Every commit | Semgrep, CodeQL |
| DAST | Weekly | OWASP ZAP |
| Dependency scan | Daily | Snyk, Dependabot |
| Container scan | Every build | Trivy |
| Secrets scan | Every commit | TruffleHog |

### Penetration Testing

- Annual third-party pentest
- Quarterly automated scanning
- Bug bounty program (post-launch)

### Security Checklist

```markdown
## Pre-Release Security Checklist

### Authentication
- [ ] All endpoints require authentication
- [ ] API keys are properly scoped
- [ ] MFA works correctly
- [ ] Session timeout implemented

### Authorization
- [ ] RBAC enforced on all resources
- [ ] Users can only access own data
- [ ] Admin actions require elevated privileges

### Data Protection
- [ ] Data encrypted at rest
- [ ] Data encrypted in transit
- [ ] PII properly handled
- [ ] Deletion fully removes data

### Input Validation
- [ ] All inputs validated
- [ ] File uploads sanitized
- [ ] SQL injection prevented
- [ ] XSS prevented

### Infrastructure
- [ ] No unnecessary ports open
- [ ] Security groups configured
- [ ] Secrets in Vault
- [ ] Logs don't contain secrets

### Monitoring
- [ ] Security events logged
- [ ] Alerts configured
- [ ] Dashboards operational
```

## Go Agent Security

### Binary Security

```go
// Compile with security flags
// CGO_ENABLED=0 GOOS=linux go build -ldflags="-s -w" -trimpath

// Certificate pinning
func createHTTPClient() *http.Client {
    certPool := x509.NewCertPool()
    certPool.AppendCertsFromPEM(embeddedCert)

    return &http.Client{
        Transport: &http.Transport{
            TLSClientConfig: &tls.Config{
                RootCAs:    certPool,
                MinVersion: tls.VersionTLS13,
            },
        },
    }
}
```

### Local Data Protection

```go
// Encrypt local model cache
func encryptModel(data []byte, key []byte) ([]byte, error) {
    block, err := aes.NewCipher(key)
    if err != nil {
        return nil, err
    }

    gcm, err := cipher.NewGCM(block)
    if err != nil {
        return nil, err
    }

    nonce := make([]byte, gcm.NonceSize())
    if _, err := io.ReadFull(rand.Reader, nonce); err != nil {
        return nil, err
    }

    return gcm.Seal(nonce, nonce, data, nil), nil
}
```

### Secure Configuration

```yaml
# Default secure configuration
security:
  # Bind to localhost only by default
  bind_address: "127.0.0.1"

  # TLS required for external connections
  tls:
    enabled: true
    cert_file: ""  # Auto-generated if empty
    key_file: ""

  # API authentication
  auth:
    required: true
    api_key_file: "~/.ads-agent/api_key"

  # Restrict file access
  allowed_paths:
    - "~/.ads-agent/models"
    - "~/.ads-agent/cache"
```
