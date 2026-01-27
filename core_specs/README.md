# Core Specifications

**Status: CONSTITUTIONAL**

This directory contains the authoritative specifications for the MEWS-FIN system. These files define the **system contract** that all services must follow.

## Purpose

The core-specs directory serves as:

1. **Single Source of Truth** — All feature definitions, data schemas, temporal rules, and risk score semantics are defined here
2. **Contract Layer** — Services consume these specs; they do not define their own interpretations
3. **Stability Anchor** — Changes to specs require versioned migrations and are treated as breaking changes

## Files

| File | Purpose | Mutability |
|------|---------|------------|
| `features.yaml` | Feature contracts for all risk indicators | Versioned changes only |
| `datasets.yaml` | Dataset schemas for ingestion services | Versioned changes only |
| `time_alignment.yaml` | Temporal rules (UTC, alignment, lookahead prevention) | **IMMUTABLE** |
| `risk_score.yaml` | Risk score semantics and regime definitions | Versioned changes only |

## Why Specs-First?

### Explainability

Every MEWS-FIN output must be traceable to these specifications. If a risk score is 0.62, we can explain:

- Which features contributed (defined in `features.yaml`)
- What data was used (defined in `datasets.yaml`)
- How time alignment was handled (defined in `time_alignment.yaml`)
- What 0.62 means (defined in `risk_score.yaml`)

### Reproducibility

Backtesting requires that historical computations can be exactly reproduced. Spec drift breaks this:

- If feature definitions change silently, historical scores become incomparable
- If temporal rules are violated, lookahead bias corrupts results
- If score semantics shift, regime interpretations become meaningless

### Service Decoupling

Services should be independently testable and replaceable:

- **Ingestion services** read `datasets.yaml` to know what to produce
- **Feature services** read `features.yaml` to know what to compute
- **Scoring services** read `risk_score.yaml` to know output requirements
- All services read `time_alignment.yaml` for temporal coordination

## Rules for Consumers

### Ingestion Services

```
MUST produce data conforming to datasets.yaml schemas
MUST respect timestamp format and timezone rules
MUST NOT add fields not in schema without spec update
```

### Feature Services

```
MUST compute features exactly as defined in features.yaml
MUST respect normalization, window, and frequency specs
MUST NOT use lookahead (enforced by time_alignment.yaml)
MUST propagate nulls according to missing data rules
```

### Scoring Services

```
MUST produce scores in [0.0, 1.0] range
MUST provide explainability outputs per risk_score.yaml
MUST include required metadata
MUST calibrate against historical anchors
```

## Changing Specs

Spec changes are **breaking changes**. Follow this process:

1. **Propose** — Document rationale for change
2. **Version** — Increment `schema_version` in affected file
3. **Migrate** — Update all consuming services
4. **Validate** — Ensure historical reproducibility is maintained or explicitly broken
5. **Document** — Record change in changelog

### Version Format

```yaml
schema_version: "1.0"
last_updated: "2026-01-28"
```

Major version changes (1.0 → 2.0) indicate breaking changes.
Minor version changes (1.0 → 1.1) indicate additive changes.

## Immutability of time_alignment.yaml

The `time_alignment.yaml` file is marked **IMMUTABLE** because:

- Temporal rules are foundational to all other specs
- Changing alignment rules invalidates all historical data
- Lookahead prevention is a hard constraint, not a preference

If temporal rules must change, it constitutes a new system version, not a spec update.
