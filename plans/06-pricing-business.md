# Pricing & Business Model

## Revenue Model Overview

Three revenue streams:
1. **Training Fees** - One-time cost to train models
2. **Hosting Fees** - Monthly for cloud-served models
3. **Inference Fees** - Per-request for cloud inference

## Pricing Philosophy

- **Transparent** - Users see exact cost before committing
- **Usage-Based** - Pay for what you use
- **Predictable** - No surprise charges
- **Tiered** - Compliance/industry-specific pricing

## The 7 Plans

### 1. Basic (Hobbyist/Testing)

**$0/month + Pay-per-use**

| Feature | Limit |
|---------|-------|
| Training | Up to 100MB data |
| Models | 2 active models |
| Inference | 1,000 requests/month free |
| Storage | 500MB |
| Support | Community forums |
| Data Retention | 30 days |

**Training Pricing:**
- $5 base fee per training job
- $0.50/MB of training data
- $2.50/GPU-hour

**Use Case:** Developers testing the platform, small personal projects

---

### 2. Indie Developer ($29/month)

| Feature | Limit |
|---------|-------|
| Training | Up to 1GB data |
| Models | 5 active models |
| Inference | 10,000 requests/month included |
| Storage | 5GB |
| Support | Email (48hr response) |
| Data Retention | 90 days |
| Extras | API access, webhooks |

**Training Pricing:**
- No base fee
- $0.40/MB of training data
- $2.00/GPU-hour

**Overage:**
- $0.002 per inference request beyond included

**Use Case:** Solo developers, small apps, side projects

---

### 3. Professional ($149/month)

| Feature | Limit |
|---------|-------|
| Training | Up to 10GB data |
| Models | 20 active models |
| Inference | 100,000 requests/month included |
| Storage | 50GB |
| Support | Email (24hr), Chat |
| Data Retention | 1 year |
| Extras | Priority training queue, custom domains |

**Training Pricing:**
- No base fee
- $0.30/MB of training data
- $1.50/GPU-hour

**Overage:**
- $0.001 per inference request beyond included

**Use Case:** Startups, growing applications, professional use

---

### 4. Legal & Compliance ($499/month)

| Feature | Limit |
|---------|-------|
| Training | Up to 50GB data |
| Models | 50 active models |
| Inference | 500,000 requests/month included |
| Storage | 250GB |
| Support | Priority email (4hr), Chat, Phone |
| Data Retention | 7 years (configurable) |
| Compliance | SOC 2 Type II, audit logs |
| Extras | SSO, dedicated support rep |

**Training Pricing:**
- No base fee
- $0.25/MB of training data
- $1.25/GPU-hour

**Features:**
- Full audit trail
- Data access logging
- Export for legal discovery
- Document retention policies
- Chain of custody tracking

**Use Case:** Law firms, compliance teams, legal tech

---

### 5. Medical / HIPAA ($999/month)

| Feature | Limit |
|---------|-------|
| Training | Up to 100GB data |
| Models | 100 active models |
| Inference | 1,000,000 requests/month included |
| Storage | 500GB |
| Support | 24/7 priority, dedicated CSM |
| Data Retention | Configurable (up to perpetual) |
| Compliance | HIPAA, SOC 2, HITRUST ready |
| Security | End-to-end encryption, BAA included |

**Training Pricing:**
- No base fee
- $0.20/MB of training data
- $1.00/GPU-hour

**Features:**
- Business Associate Agreement (BAA)
- PHI data handling
- HIPAA-compliant infrastructure
- Breach notification procedures
- De-identification tools
- Access controls & audit logs

**Use Case:** Healthcare providers, medical research, health tech

---

### 6. Government ($1,499/month)

| Feature | Limit |
|---------|-------|
| Training | Up to 200GB data |
| Models | 200 active models |
| Inference | 2,000,000 requests/month included |
| Storage | 1TB |
| Support | 24/7 priority, dedicated team |
| Data Retention | Configurable per regulation |
| Compliance | FedRAMP Moderate, FISMA, CJIS |
| Security | US-only data residency, cleared staff |

**Training Pricing:**
- No base fee
- $0.18/MB of training data
- $0.90/GPU-hour

**Features:**
- FedRAMP Moderate authorization path
- FISMA compliance controls
- CJIS security policy alignment
- US citizen staff only
- Air-gapped deployment option
- IL4/IL5 capable infrastructure

**Use Case:** Federal agencies, state/local government, contractors

---

### 7. Finance ($1,999/month)

| Feature | Limit |
|---------|-------|
| Training | Up to 500GB data |
| Models | Unlimited |
| Inference | 5,000,000 requests/month included |
| Storage | 2TB |
| Support | 24/7 white-glove, dedicated team |
| Data Retention | Configurable (regulatory compliance) |
| Compliance | SOC 2, PCI-DSS ready, SEC/FINRA |
| SLA | 99.99% uptime, <100ms P99 latency |

**Training Pricing:**
- No base fee
- $0.15/MB of training data
- $0.80/GPU-hour

**Features:**
- Financial data handling protocols
- SEC/FINRA compliance support
- PCI-DSS ready infrastructure
- Market data integration
- Real-time inference guarantees
- Disaster recovery (multi-region)

**Use Case:** Banks, hedge funds, fintech, insurance

---

### Custom / Enterprise

**Contact Sales**

For organizations that need:
- Custom compliance frameworks
- On-premises deployment
- Dedicated infrastructure
- Volume discounts beyond standard tiers
- Multi-region requirements
- Custom integrations
- White-label options

## Pricing Calculator Logic

### Training Cost Formula

```python
def calculate_training_cost(
    data_size_mb: float,
    model_size: str,  # tiny, small, medium, large
    quality_preset: str,  # quick, balanced, thorough, maximum
    plan_tier: str,
) -> TrainingEstimate:

    # Base rates (per MB of training data)
    base_rates = {
        'basic': 0.50,
        'indie': 0.40,
        'professional': 0.30,
        'legal': 0.25,
        'medical': 0.20,
        'government': 0.18,
        'finance': 0.15,
    }

    # GPU hour rates
    gpu_rates = {
        'basic': 2.50,
        'indie': 2.00,
        'professional': 1.50,
        'legal': 1.25,
        'medical': 1.00,
        'government': 0.90,
        'finance': 0.80,
    }

    # Model size multipliers (affects GPU hours)
    size_multipliers = {
        'tiny': 0.5,
        'small': 1.0,
        'medium': 2.5,
        'large': 5.0,
    }

    # Quality preset multipliers
    quality_multipliers = {
        'quick': 1.0,
        'balanced': 3.0,
        'thorough': 6.0,
        'maximum': 15.0,
    }

    # Calculate components
    data_cost = data_size_mb * base_rates[plan_tier]

    # Estimate GPU hours
    base_gpu_hours = data_size_mb / 100  # ~1 hour per 100MB baseline
    gpu_hours = (
        base_gpu_hours
        * size_multipliers[model_size]
        * quality_multipliers[quality_preset]
    )

    compute_cost = gpu_hours * gpu_rates[plan_tier]

    # Base fee (only for basic tier)
    base_fee = 5.0 if plan_tier == 'basic' else 0.0

    total = base_fee + data_cost + compute_cost

    return TrainingEstimate(
        base_fee=base_fee,
        data_cost=data_cost,
        compute_cost=compute_cost,
        total=total,
        estimated_gpu_hours=gpu_hours,
        estimated_time_minutes=int(gpu_hours * 60),
    )
```

### Example Calculations

#### Scenario 1: Small Startup (Indie Plan)
- 500MB training data
- Small model
- Balanced quality

```
Data cost: 500 × $0.40 = $200
GPU hours: (500/100) × 1.0 × 3.0 = 15 hours
Compute: 15 × $2.00 = $30
Total: $230
```

#### Scenario 2: Law Firm (Legal Plan)
- 5GB (5,000MB) training data
- Medium model
- Thorough quality

```
Data cost: 5,000 × $0.25 = $1,250
GPU hours: (5,000/100) × 2.5 × 6.0 = 750 hours
Compute: 750 × $1.25 = $937.50
Total: $2,187.50
```

#### Scenario 3: Hospital (Medical Plan)
- 20GB (20,000MB) training data
- Large model
- Maximum quality

```
Data cost: 20,000 × $0.20 = $4,000
GPU hours: (20,000/100) × 5.0 × 15.0 = 15,000 hours
Compute: 15,000 × $1.00 = $15,000
Total: $19,000
```

## Cloud Hosting Pricing

For models served via aiserve.farm (not local agent):

### Model Hosting

| Model Size | Monthly Cost | Included Requests |
|------------|--------------|-------------------|
| Tiny (<50MB) | $10/month | 10,000 |
| Small (<300MB) | $25/month | 25,000 |
| Medium (<1GB) | $75/month | 75,000 |
| Large (<2GB) | $150/month | 150,000 |

### Inference Pricing (Beyond Included)

| Tier | Cost per 1K Requests |
|------|---------------------|
| Basic | $2.00 |
| Indie | $1.50 |
| Professional | $1.00 |
| Legal+ | $0.50 |

## Cost Optimization Features

### Auto-Scaling
- Scale down during low usage
- Scale up for burst traffic
- Pay only for actual usage

### Caching
- Response caching reduces inference costs
- Configurable TTL
- Cache hit doesn't count against quota

### Batch Processing
- Bulk inference at discounted rates
- Up to 50% savings on batch jobs

### Reserved Capacity
- Commit to monthly usage for discounts
- 20% off with 6-month commitment
- 35% off with annual commitment

## Payment & Billing

### Payment Methods
- Credit/debit cards (Stripe)
- ACH/wire transfer (enterprise)
- Purchase orders (government/enterprise)
- Invoicing (net-30 for qualified accounts)

### Billing Cycle
- Monthly billing on anniversary date
- Usage calculated at end of billing period
- Overage charges billed monthly
- Training jobs charged immediately

### Credits System
- Pre-purchase credits at discount
- Credits never expire
- Referral credits ($100 per referred customer)

## Competitive Positioning

### vs. AWS SageMaker
- **Simpler**: No ML expertise required
- **Cheaper**: No infrastructure management overhead
- **Faster**: Pre-built pipelines vs. DIY

### vs. OpenAI Fine-tuning
- **Private**: Your data stays yours
- **Flexible**: More model architectures
- **Ownership**: Download and run models anywhere

### vs. Hugging Face AutoTrain
- **Full Service**: We handle everything
- **Compliance**: Industry-specific certifications
- **Support**: White-glove service options

## Revenue Projections

### Year 1 Targets

| Quarter | Customers | MRR | Training Revenue |
|---------|-----------|-----|------------------|
| Q1 | 50 | $15,000 | $25,000 |
| Q2 | 150 | $45,000 | $75,000 |
| Q3 | 400 | $120,000 | $150,000 |
| Q4 | 800 | $250,000 | $300,000 |

### Customer Mix Target (End of Year 1)

| Plan | Customers | % of MRR |
|------|-----------|----------|
| Basic | 400 | 5% |
| Indie | 250 | 15% |
| Professional | 100 | 25% |
| Legal | 30 | 20% |
| Medical | 15 | 15% |
| Government | 3 | 10% |
| Finance | 2 | 10% |

## Monetization Timeline

### Phase 1: MVP (Months 1-3)
- Basic + Indie plans only
- Local serving (Go agent)
- Training fees only

### Phase 2: Growth (Months 4-6)
- Add Professional plan
- Cloud hosting option
- Inference pricing

### Phase 3: Enterprise (Months 7-12)
- Legal, Medical, Government, Finance plans
- Compliance certifications
- Enterprise features

### Phase 4: Scale (Year 2+)
- White-label licensing
- Partner program
- Marketplace for pre-trained models
