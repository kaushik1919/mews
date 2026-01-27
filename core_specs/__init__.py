"""
MEWS Core Specifications Package.

This package contains YAML specification files that define the
contracts for datasets, features, time alignment, and risk scoring.

These specs are the "constitutional layer" of MEWS:
- datasets.yaml: Schema definitions for all data sources
- features.yaml: Feature definitions and metadata
- time_alignment.yaml: Rules for temporal alignment
- risk_score.yaml: Risk scoring methodology

Access specs via importlib.resources:

    from importlib import resources
    import yaml

    with resources.files("core_specs").joinpath("datasets.yaml").open("r") as f:
        schemas = yaml.safe_load(f)

No filesystem paths. Install-safe.
"""
