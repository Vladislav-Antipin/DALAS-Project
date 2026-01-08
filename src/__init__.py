"""DALAS Drug Repurposing Package.

This package provides functionality matching the logic from notebooks:
- 01_1-data_retrieval.ipynb
- 01_2-rdkit_fingerprints.ipynb
- 01_3-additional_disease_features.ipynb
- 02-data_preparation.ipynb

Usage:
    from src import run_full_pipeline
    run_full_pipeline()
    
    # Or load individual datasets:
    from src import load_drugs, load_diseases, load_indications
"""

from .data_loader import (
    load_all_processed,
    load_all_raw,
    load_drugs,
    load_diseases,
    load_trials,
    load_indications,
    load_dates,
    load_embeddings,
    load_fingerprints,
    load_extended_diseases,
)

from .pipeline import (
    run_full_pipeline,
    run_data_fetching,
    run_data_processing,
    run_embeddings,
)

from .fingerprints import generate_fingerprints
from .disease_features import generate_extended_disease_features
from .pathway_mapping import map_targets_to_pathways, compute_pathway_similarity
from .embeddings import compute_drug_disease_similarities

__all__ = [
    # Data loaders
    "load_all_processed",
    "load_all_raw",
    "load_drugs",
    "load_diseases",
    "load_trials",
    "load_indications",
    "load_dates",
    "load_embeddings",
    "load_fingerprints",
    "load_extended_diseases",
    # Pipeline functions
    "run_full_pipeline",
    "run_data_fetching",
    "run_data_processing",
    "run_embeddings",
    # Feature generation
    "generate_fingerprints",
    "generate_extended_disease_features",
    "map_targets_to_pathways",
    "compute_pathway_similarity",
    "compute_drug_disease_similarities",
]
