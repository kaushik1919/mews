# MEWS-FIN

**Market Risk Early Warning System**

A research-grade system for early detection of systemic market risk using exclusively free, public data sources.

---

## What is MEWS-FIN?

MEWS-FIN is designed to provide **early warning signals** of systemic market stress. It is:

- ✅ A **risk monitoring** system
- ✅ A **research tool** for understanding market dynamics
- ✅ An **interpretable** framework with explainable outputs
- ❌ **NOT** a trading system
- ❌ **NOT** a prediction engine
- ❌ **NOT** a black-box model

The primary output is a single **risk score** in [0, 1] that quantifies current systemic stress levels based on observable market signals.

### Data Sources

| Category | Sources | Purpose |
|----------|---------|---------|
| Market Prices | Yahoo Finance, Stooq | Volatility, drawdowns, correlations |
| Macro Rates | FRED | Credit spreads, funding stress |
| Volatility | VIX via Yahoo/Stooq | Implied volatility, fear gauge |
| News/Sentiment | Common Crawl, RSS feeds | Sentiment via FinBERT |

### Rationale

1. **Reproducibility** — Anyone can reproduce the analysis
2. **Transparency** — No hidden data advantages
3. **Accessibility** — Research-friendly, no cost barriers
4. **Independence** — No vendor dependencies

## Architecture

MEWS-FIN follows a **logical microservices, physical monolith** architecture:

```
┌─────────────────────────────────────────────────────────┐
│                     MEWS-FIN                            │
├─────────────────────────────────────────────────────────┤
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Ingestion│→ │ Features │→ │ Scoring  │→ │ Output  │ │
│  │ Service  │  │ Service  │  │ Service  │  │ Service │ │
│  └──────────┘  └──────────┘  └──────────┘  └─────────┘ │
│       ↑              ↑             ↑            ↑      │
│       └──────────────┴─────────────┴────────────┘      │
│                    core-specs/                         │
└─────────────────────────────────────────────────────────┘
```

- **Logical microservices** — Clear boundaries, testable in isolation
- **Physical monolith** — Simple deployment, no distributed complexity
- **Spec-driven** — All services consume `core-specs/` definitions

## Project Phases

### Phase 1: Foundation ✅ 
- Repository scaffold
- Core specifications (`core-specs/`)
- Documentation
- CI/CD pipeline

### Phase 2: Data Layer (Current)
- Ingestion services for each data source
- Data validation and quality checks
- Storage infrastructure

### Phase 3: Feature Engine (Planned)
- Feature computation services
- Temporal alignment implementation
- Feature validation

### Phase 4: Scoring (Planned)
- Heuristic scoring model (baseline)
- Explainability outputs
- Calibration against historical events

### Phase 5: Interface (Planned)
- API layer
- Monitoring dashboard
- Alert system

## Why Specs Come First

The `core-specs/` directory is the **constitutional layer** of MEWS-FIN. It defines:

- **What** features exist and how they're computed
- **What** data schemas services must produce/consume  
- **How** time alignment works (UTC, no lookahead)
- **What** the risk score means semantically

This approach ensures:

1. **Explainability** — Every output traces to specifications
2. **Reproducibility** — Historical analysis can be exactly repeated
3. **Testability** — Services can be tested against spec contracts
4. **Stability** — Meaning doesn't drift with implementation changes

See [`core-specs/README.md`](core-specs/README.md) for details.

## Quick Start

### Requirements

- Python 3.10+
- Dependencies in `pyproject.toml`

### Installation

```bash
# Clone repository
git clone https://github.com/your-org/mews-fin.git
cd mews-fin

# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate  # Unix

# Install dependencies
pip install -e ".[dev]"
```

### Validate Specs

```bash
# Check YAML syntax
yamllint core-specs/

# Run linting
ruff check .
```

## Engineering Principles

### Determinism
Same inputs → Same outputs. No hidden state, no randomness without seeds.

### Testability
Every module testable in isolation. Clear interfaces, explicit dependencies.

### Explainability
No black boxes. Every risk score comes with feature contributions.

### Correctness Over Performance
Prefer readable, correct code. Optimize only when necessary.

## Contributing

This is a research project. Contributions should:

1. Respect existing architecture and specs
2. Include tests
3. Add docstrings explaining assumptions
4. Avoid scope creep beyond MEWS-FIN's purpose

## License

MIT License — See LICENSE file.

---

**MEWS-FIN** — Interpretable early warning for systemic market risk.
