# DALAS Data Pipeline

This directory contains modular Python scripts for fetching and processing autoimmune disease drug data from various sources.

**The scripts match the logic from the Jupyter notebooks** so users can run either the notebooks or the Python scripts and get the same results.

## Structure

```
src/
├── __init__.py            # Package exports
├── config.py              # Configuration, paths, and constants
├── data_fetchers.py       # Functions to fetch raw data from APIs
├── data_processors.py     # Functions to clean and process raw data
├── data_loader.py         # Utility functions for loading data
├── embeddings.py          # Functions to generate name embeddings
├── fingerprints.py        # RDKit molecular fingerprints (notebook 01_2)
├── disease_features.py    # Additional disease features (notebook 01_3)
├── pathway_mapping.py     # Reactome pathway mapping (notebook 01_1)
└── pipeline.py            # Main orchestration script
```

## Notebook Correspondence

| Notebook | Python Module(s) |
|----------|------------------|
| `01_1-data_retrieval.ipynb` | `data_fetchers.py`, `data_processors.py`, `embeddings.py`, `pathway_mapping.py` |
| `01_2-rdkit_fingerprints.ipynb` | `fingerprints.py` |
| `01_3-additional_disease_features.ipynb` | `disease_features.py` |
| `02-data_preparation.ipynb` | `data_processors.py` |

## Data Flow

```
1. RAW DATA FETCHING (data_fetchers.py) → data/01-result/
   ├─ MeSH IDs (SPARQL)     → mesh_ids.pkl
   ├─ Drug Indications      → indications_df.pkl
   ├─ Drugs                 → drugs_df.pkl
   ├─ Mechanisms            → mechanism_df.pkl
   ├─ Targets               → targets_df.pkl
   ├─ Diseases              → diseases_df.pkl
   └─ Clinical Trials       → trials_df.pkl

2. DATA PROCESSING → data/01-result/
   ├─ Pathway Mapping       → drugs_df.pkl (with drug_pathways)
   │                        → diseases_df.pkl (with disease_pathways)
   ├─ Fingerprints (PCA)    → fingerprints_df.pkl
   ├─ Disease Features      → extended_diseases_df.pkl
   └─ Name Embeddings       → embeddings_df.pkl

3. FINAL PROCESSING → data/02-result/
   ├─ Process Drugs         → drugs_df.pkl
   ├─ Process Diseases      → diseases_df.pkl
   ├─ Process Trials        → trials_df.pkl (with first_date, last_date)
   ├─ Process Indications   → indications_df.pkl (with overall_success)
   └─ Trial Dates           → dates_df.pkl
```

## Usage

### Run the Full Pipeline

```bash
# Run everything (uses cache when available)
python -m src.pipeline

# Force re-fetch all raw data
python -m src.pipeline --force-fetch

# Force re-process all data
python -m src.pipeline --force-process

# Force everything
python -m src.pipeline --force-fetch --force-process
```

### Run Individual Steps

```bash
# Only fetch raw data
python -m src.pipeline --step fetch

# Only process data (requires raw data to exist)
python -m src.pipeline --step process

# Only generate embeddings (requires processed data to exist)
python -m src.pipeline --step embeddings
```

## Modules

### config.py

- Project paths and directories (`data/01-result/`, `data/02-result/`)
- API endpoints (MeSH SPARQL, Open Targets, ClinicalTrials.gov)
- Model configurations (PubMedBERT embeddings)
- Global parameters (NB_TOP_TARGETS=10)

### data_fetchers.py

Functions for retrieving raw data:

- `fetch_mesh_ids()` - Query MeSH SPARQL for immune system diseases
- `fetch_drug_indications()` - Get drug-disease pairs from ChEMBL
- `fetch_drugs()` - Get drug information from ChEMBL
- `fetch_mechanisms_and_targets()` - Get drug mechanisms and protein targets
- `fetch_disease_info()` - Get disease data from Open Targets with SwissProt targets
- `fetch_clinical_trials()` - Get trial data from ClinicalTrials.gov

### data_processors.py

Functions for cleaning and transforming data:

- `process_drugs_data()` - Clean drugs, extract properties, add target dictionaries
- `process_diseases_data()` - Process disease information
- `process_trials_data()` - Extract trial metadata with first_date/last_date and classify success
- `process_indications_data()` - Combine drug-disease pairs with overall_success labels

### fingerprints.py

Functions for molecular fingerprints (matches notebook 01_2):

- `generate_fingerprints()` - Generate Morgan fingerprints with PCA reduction

### disease_features.py

Functions for additional disease features (matches notebook 01_3):

- `fetch_mesh_ontology_features()` - Get MeSH tree numbers and depths
- `fetch_disease_categories()` - Get disease category mappings
- `generate_extended_disease_features()` - One-hot encode categories, add prevalence

### pathway_mapping.py

Functions for Reactome pathway mapping (matches notebook 01_1):

- `map_targets_to_pathways()` - Map UniProt targets to Reactome pathways
- `generate_pathway_embeddings()` - Generate TF-IDF pathway embeddings
- `compute_pathway_similarity()` - Compute Jaccard similarity between pathway lists

### embeddings.py

Functions for semantic analysis:

- `compute_drug_disease_similarities()` - Generate name similarity scores using PubMedBERT
- `generate_text_embeddings()` - Create embeddings for arbitrary text

### pipeline.py

Main orchestration:

- `run_data_fetching()` - Execute all fetching steps
- `run_data_processing()` - Execute all processing steps (including fingerprints, pathways, features)
- `run_embeddings()` - Generate name embeddings
- `run_full_pipeline()` - Execute complete pipeline

## Data Outputs

### Raw Data (data/01-result/)

Directly fetched from APIs with initial processing:

- **drugs_df.pkl**: Drug information with targets and drug_pathways
- **diseases_df.pkl**: Disease information with disease_targets and disease_pathways
- **fingerprints_df.pkl**: Morgan fingerprints reduced to 100 PCA components
- **embeddings_df.pkl**: Drug-disease name similarity scores
- **extended_diseases_df.pkl**: Disease features with ontology and categories

### Processed Data (data/02-result/)

Cleaned and structured for analysis:

- **drugs_df.pkl**: Final drug features with pathway embeddings
- **diseases_df.pkl**: Final disease features with pathway embeddings
- **trials_df.pkl**: Clinical trials with first_date, last_date, and success classification
- **indications_df.pkl**: Drug-disease pairs with overall_success labels
- **dates_df.pkl**: First trial dates for time-based train/test splits

## Dependencies

Required packages:

- pandas
- numpy
- requests
- lxml
- chembl_webresource_client
- bioservices
- sentence_transformers
- rdkit
- scikit-learn
- python-dotenv

## Notes

- All functions support caching - data is only re-fetched when `force=True`
- Raw data is saved to `data/01-result/` (matching notebook outputs)
- Processed data is saved to `data/02-result/` (matching notebook outputs)
- The pipeline respects API rate limits (e.g., ClinicalTrials.gov: 10 req/sec)
- UniProt ID mapping can be slow for large datasets
- MeSH IDs are fetched via SPARQL query (not web scraping)
