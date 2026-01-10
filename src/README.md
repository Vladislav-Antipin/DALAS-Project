# DALAS Data Pipeline

Modular Python package for fetching and processing autoimmune disease drug data.

## Structure

```text
src/
├── __init__.py            # Package exports
├── config.py              # Configuration, paths, and constants
├── data_fetchers.py       # Functions to fetch raw data from APIs
├── data_processors.py     # Functions to clean and process raw data
├── data_loader.py         # Convenience functions to load cached data
├── embeddings.py          # Functions to generate embeddings
└── pipeline.py            # Main orchestration script
```

## Usage

### Run the Full Pipeline

```bash
# Run everything (uses cache when available)
uv run python -m src.pipeline

# Force re-fetch all raw data
uv run python -m src.pipeline --force-fetch

# Force re-process all data
uv run python -m src.pipeline --force-process

# Force everything
uv run python -m src.pipeline --force-fetch --force-process
```

### Run Individual Steps

```bash
# Only fetch raw data
uv run python -m src.pipeline --step fetch

# Only process data (requires raw data to exist)
uv run python -m src.pipeline --step process

# Only generate embeddings (requires processed data to exist)
uv run python -m src.pipeline --step embeddings
```

### Load Data in Python

```python
from src import load_drugs, load_diseases, load_indications, load_embeddings

# Load individual dataframes
drugs_df = load_drugs()
diseases_df = load_diseases()

# Or load everything at once
from src import load_all_processed
data = load_all_processed()
```

## Data Flow

```text
1. RAW DATA FETCHING (data_fetchers.py)
   ├─ MeSH IDs          → data/raw/mesh_ids.pkl
   ├─ Drug Indications  → data/raw/indications_df.pkl
   ├─ Drugs             → data/raw/drugs_df.pkl
   ├─ Mechanisms        → data/raw/mechanism_df.pkl
   ├─ Targets           → data/raw/targets_df.pkl
   ├─ Diseases          → data/raw/diseases_df.pkl
   └─ Clinical Trials   → data/raw/trials_df.pkl

2. DATA PROCESSING (data_processors.py)
   ├─ Process Drugs     → data/processed/final_drugs_df.pkl
   ├─ Process Diseases  → data/processed/final_diseases_df.pkl
   ├─ Process Trials    → data/processed/final_trials_df.pkl
   └─ Process Indications → data/processed/final_indications_df.pkl

3. EMBEDDINGS (embeddings.py)
   └─ Drug-Disease Similarities → data/processed/embeddings_df.pkl
```

## Modules

### config.py

Project configuration:
- `PROJECT_ROOT`, `DATA_DIR`, `RAW_DATA_DIR`, `PROCESSED_DATA_DIR`, `RESULTS_DIR`
- API endpoints: `CHEMBL_API`, `OPENFDA_API`, `PUBMED_API`, `OT_URL`, `NCT_URL`, `MESH_URL`
- Model settings: `ST_MODEL` (PubMedBERT)
- Parameters: `NB_TOP_TARGETS`, `NB_EVIDENCES`

### data_fetchers.py

Functions for retrieving raw data:
- `fetch_mesh_ids()` — Scrape MeSH IDs for autoimmune diseases from NCBI
- `fetch_drug_indications()` — Get drug-disease pairs from ChEMBL
- `fetch_drugs()` — Get drug information from ChEMBL
- `fetch_mechanisms_and_targets()` — Get drug mechanisms and protein targets (with UniProt mapping)
- `fetch_disease_info()` — Get disease data and associated targets from Open Targets
- `fetch_clinical_trials()` — Get trial data from ClinicalTrials.gov

### data_processors.py

Functions for cleaning and transforming data:
- `process_drugs_data()` — Filter available drugs, extract properties, add target dictionaries
- `process_diseases_data()` — Process disease information
- `process_trials_data()` — Extract trial metadata and classify success based on p-values
- `process_indications_data()` — Combine drug-disease pairs with clinical trial evidence

### data_loader.py

Convenience functions for loading cached data:
- `load_all_processed()` — Load all processed dataframes as a dictionary
- `load_all_raw()` — Load all raw dataframes as a dictionary
- `load_drugs()`, `load_diseases()`, `load_trials()`, `load_indications()`, `load_embeddings()`

### embeddings.py

Functions for semantic analysis:
- `compute_drug_disease_similarities()` — Compute cosine similarity between drug and disease names using PubMedBERT
- `generate_text_embeddings()` — Generate embeddings for arbitrary text

### pipeline.py

Main orchestration:
- `run_data_fetching()` — Execute all fetching steps
- `run_data_processing()` — Execute all processing steps
- `run_embeddings()` — Generate embeddings
- `run_full_pipeline()` — Execute complete pipeline

## Notes

- All functions support caching — data is only re-fetched when `force=True`
- The pipeline respects API rate limits (ClinicalTrials.gov: 10 req/sec)
- UniProt ID mapping can be slow for large datasets
