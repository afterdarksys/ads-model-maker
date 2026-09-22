# MVP Roadmap

## MVP Definition

The Minimum Viable Product delivers core value: **Users upload data, get a trained model they can use immediately.**

### MVP Scope

**In Scope:**
- PDF and TXT document ingestion
- Text classification models
- Go agent for local serving
- Basic web dashboard
- Training status monitoring
- Model download
- Basic API

**Out of Scope (Post-MVP):**
- Image/vision models
- Database connectors
- Cloud hosting
- MCP integration
- Agent capabilities
- Compliance certifications
- Advanced analytics

## Development Phases

### Phase 1: Foundation

**Goal:** Core infrastructure and basic ingestion

#### Python Backend

| Task | Description |
|------|-------------|
| Project setup | Python project structure, dependencies, CI/CD |
| Data service | FastAPI service for upload handling |
| PDF processor | Extract text from PDFs using pdfplumber |
| TXT processor | Handle plain text files |
| Chunker | Semantic text chunking |
| Storage adapter | S3/OCI object storage integration |
| Database schema | PostgreSQL tables for users, jobs, models |

#### Go Agent (Parallel)

| Task | Description |
|------|-------------|
| Project setup | Go project structure, build scripts |
| HTTP server | Fiber-based API server |
| Health endpoints | /health, /ready endpoints |
| Config loading | YAML configuration parsing |
| Build pipeline | Cross-compilation scripts |

#### Infrastructure

| Task | Description |
|------|-------------|
| Docker compose | Local development environment |
| PostgreSQL setup | Database with initial schema |
| Redis setup | Task queue infrastructure |
| Object storage | MinIO for local dev, OCI for prod |

**Deliverable:** Upload a PDF, see it processed and chunked

---

### Phase 2: Training Pipeline

**Goal:** Train classification models from processed data

#### Training System

| Task | Description |
|------|-------------|
| Training service | Celery workers for training jobs |
| Base model loader | Load distilbert from HF |
| Dataset builder | Convert chunks to HF Dataset |
| Fine-tuning | LoRA-based classification training |
| Evaluation | Accuracy, F1, confusion matrix |
| Model export | Save as safetensors |
| Job status API | Real-time training progress |

#### User Interface

| Task | Description |
|------|-------------|
| Dashboard scaffolding | React/Next.js basic layout |
| Data source list | View uploaded data sources |
| Training form | Configure and start training |
| Progress display | Real-time training status |
| Model list | View trained models |

**Deliverable:** Upload data → Configure training → Get trained model

---

### Phase 3: Model Serving

**Goal:** Use trained models for inference

#### Go Agent Inference

| Task | Description |
|------|-------------|
| Model downloader | Fetch models from storage |
| ONNX integration | Load and run ONNX models |
| Inference API | /predict, /classify endpoints |
| Response caching | BigCache integration |
| Model management | Load/unload models |

#### Backend Support

| Task | Description |
|------|-------------|
| Model export to ONNX | Convert PyTorch to ONNX |
| Download endpoint | Signed URLs for model download |
| Usage tracking | Log inference requests |

#### User Interface

| Task | Description |
|------|-------------|
| Model detail view | Metrics, config, usage |
| Download button | Get model files |
| Playground | Test model in browser |
| Agent setup guide | Installation instructions |

**Deliverable:** Download model → Install agent → Make predictions

---

### Phase 4: Polish & Launch

**Goal:** Production-ready MVP

#### Quality & Testing

| Task | Description |
|------|-------------|
| Unit tests | 80%+ coverage on critical paths |
| Integration tests | End-to-end workflow tests |
| Load testing | Verify performance targets |
| Security audit | Basic security review |

#### Documentation

| Task | Description |
|------|-------------|
| User guide | How to use the platform |
| API reference | OpenAPI spec + examples |
| Agent docs | Installation and configuration |
| Troubleshooting | Common issues and solutions |

#### Operations

| Task | Description |
|------|-------------|
| Monitoring | Prometheus + Grafana dashboards |
| Alerting | PagerDuty/Slack alerts |
| Logging | Centralized log aggregation |
| Backups | Database and storage backups |

#### Launch Prep

| Task | Description |
|------|-------------|
| Landing page | Marketing site |
| Pricing page | Plan comparison |
| Signup flow | Stripe integration |
| Onboarding | First-user experience |

**Deliverable:** Public launch with paying customers

---

## Post-MVP Phases

### Phase 5: Extended Formats

- DOCX, RTF, HTML ingestion
- CSV/JSON structured data
- Image ingestion (basic)
- Database connectors (PostgreSQL)

### Phase 6: More Model Types

- Q&A models
- Embedding models
- Summarization models
- Named entity recognition

### Phase 7: Cloud Serving

- Hosted inference option
- Auto-scaling infrastructure
- Per-request billing
- Geographic distribution

### Phase 8: Advanced Features

- MCP server integration
- Autonomous agents
- GoLearn native models
- Vision models (classification, OCR)

### Phase 9: Compliance

- SOC 2 Type II certification
- HIPAA compliance
- FedRAMP readiness
- PCI-DSS alignment

### Phase 10: Enterprise

- SSO/SAML integration
- Role-based access control
- Multi-tenant isolation
- White-label options

---

## Technical Milestones

### Milestone 1: First Trained Model
- [ ] User can upload a PDF
- [ ] Text is extracted and chunked
- [ ] Classification model trains successfully
- [ ] Model file is downloadable

### Milestone 2: First Inference
- [ ] Go agent runs locally
- [ ] Model loads successfully
- [ ] /predict endpoint works
- [ ] Response time < 100ms

### Milestone 3: First Paying Customer
- [ ] Signup flow works
- [ ] Payment processes successfully
- [ ] Training job completes
- [ ] Customer gets working model

### Milestone 4: Ten Active Users
- [ ] Platform handles concurrent users
- [ ] No critical bugs in production
- [ ] Support tickets manageable
- [ ] Positive user feedback

---

## Risk Mitigation

### Technical Risks

| Risk | Mitigation |
|------|------------|
| Training quality inconsistent | Extensive validation, sensible defaults |
| Go agent compatibility issues | Test on Windows/macOS/Linux CI |
| Performance bottlenecks | Load test early, profile regularly |
| Data loss | Automated backups, replication |

### Business Risks

| Risk | Mitigation |
|------|------------|
| Low conversion rate | Focus on UX, clear value prop |
| Competition | Speed to market, niche focus |
| Pricing wrong | Start low, adjust based on feedback |
| Support overwhelm | Self-service docs, FAQ, chatbot |

---

## Resource Requirements

### Team (MVP Phase)

| Role | Allocation |
|------|------------|
| Backend Engineer (Python) | 1 FTE |
| Go Engineer | 0.5 FTE |
| Frontend Engineer | 0.5 FTE |
| DevOps/Infra | 0.25 FTE |

### Infrastructure (MVP)

| Resource | Specification | Monthly Cost |
|----------|---------------|--------------|
| Training GPU | 1x A100 40GB (on-demand) | ~$500-1000 |
| API Servers | 2x 4vCPU/8GB | ~$100 |
| Database | Managed PostgreSQL | ~$50 |
| Object Storage | 100GB | ~$25 |
| Redis | Managed Redis | ~$30 |

**Estimated MVP Monthly Infra Cost: $700-1,200**

---

## Definition of Done

### For MVP Launch:

- [ ] User can sign up and pay
- [ ] User can upload PDF documents
- [ ] User can configure and start training
- [ ] User can monitor training progress
- [ ] User can download trained model
- [ ] User can run Go agent locally
- [ ] User can make inference requests
- [ ] Documentation covers all features
- [ ] System handles 100 concurrent users
- [ ] 99.9% uptime for 2 weeks pre-launch
- [ ] Zero critical security vulnerabilities
- [ ] Support channels operational

---

## Success Metrics (MVP)

| Metric | Target |
|--------|--------|
| Signup to trained model | < 30 minutes |
| Training success rate | > 95% |
| Agent install success | > 90% |
| First inference success | > 98% |
| NPS score | > 40 |
| Support ticket volume | < 1 per 10 users |

---

## Launch Checklist

### Pre-Launch (1 week before)

- [ ] All critical bugs fixed
- [ ] Documentation complete
- [ ] Support playbook ready
- [ ] Monitoring dashboards live
- [ ] Alerting configured
- [ ] Backup/restore tested
- [ ] Load test passed
- [ ] Security checklist complete

### Launch Day

- [ ] DNS/CDN configured
- [ ] Payment processing verified
- [ ] Email notifications working
- [ ] Social media announcements ready
- [ ] Team on standby for issues
- [ ] Rollback plan documented

### Post-Launch (1 week after)

- [ ] Monitor error rates
- [ ] Address user feedback
- [ ] Fix any critical issues
- [ ] Gather testimonials
- [ ] Plan next iteration
