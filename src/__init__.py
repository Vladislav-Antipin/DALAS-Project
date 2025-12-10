"""DALAS Drug Repurposing Package."""

from typing import Any

from .data_loader import (
    load_all_processed,
    load_all_raw,
    load_drugs,
    load_diseases,
    load_trials,
    load_indications,
    load_embeddings,
    load_reactome_map,
)

from .model_training import (
    run_model_training,
    predict_new_pairs,
)

from .feature_engineering import (
    compute_fingerprints,
    compute_pathway_features,
    engineer_labels,
)

from .data_merger import merge_all_features

_PIPELINE_EXPORTS = {
    "run_full_pipeline",
    "run_data_fetching",
    "run_data_processing",
    "run_embeddings",
    "run_features",
    "run_merge",
    "run_training",
}

def __getattr__(name: str) -> Any:
    if name in _PIPELINE_EXPORTS:
        from . import pipeline as _pipeline

        return getattr(_pipeline, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    # Data loaders
    "load_all_processed",
    "load_all_raw",
    "load_drugs",
    "load_diseases",
    "load_trials",
    "load_indications",
    "load_embeddings",
    "load_reactome_map",
    # Pipeline functions
    *_PIPELINE_EXPORTS,
    # Feature engineering
    "compute_fingerprints",
    "compute_pathway_features",
    "engineer_labels",
    "merge_all_features",
    # Model training
    "run_model_training",
    "predict_new_pairs",
]
