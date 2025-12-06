# DALAS Data Pipeline

This directory contains modular Python scripts for fetching and processing immune system disease drug data from various sources.

## Structure

```
src/
├── config.py              # Configuration, paths, and constants
├── data_fetchers.py       # Functions to fetch raw data from APIs
├── data_processors.py     # Functions to clean and process raw data
├── embeddings.py          # Functions to generate embeddings
├── data_loader.py         # Utility functions for loading saved data
├── pipeline.py            # Main orchestration script
└── __init__.py            # Package exports
```

## Data Flow

```
1. RAW DATA FETCHING (data_fetchers.py)
   ├─ MeSH IDs          → data/raw/mesh_ids.pkl         (via SPARQL query)
   ├─ Drug Indications  → data/raw/indications_df.pkl   (from ChEMBL)
   ├─ Drugs             → data/raw/drugs_df.pkl         (from ChEMBL)
   ├─ Mechanisms        → data/raw/mechanism_df.pkl     (from ChEMBL)
   ├─ Targets           → data/raw/targets_df.pkl       (from ChEMBL)
   ├─ Diseases          → data/raw/diseases_df.pkl      (from Open Targets)
   ├─ Clinical Trials   → data/raw/trials_df.pkl        (from ClinicalTrials.gov)
   └─ Reactome Map      → data/raw/reactome_map.pkl     (from UniProt)

2. DATA PROCESSING (data_processors.py)
   ├─ Process Drugs     → data/processed/final_drugs_df.pkl     (+drug_pathways)
   ├─ Process Diseases  → data/processed/final_diseases_df.pkl  (+disease_pathways)
   ├─ Process Trials    → data/processed/final_trials_df.pkl    (+flattened structure)
   └─ Process Indications → data/processed/final_indications_df.pkl (+trial counts)

3. EMBEDDINGS (embeddings.py)
   └─ Drug-Disease Similarities → data/processed/embeddings_df.pkl
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
- Project paths and directories
- API endpoints and URLs
- Model configurations
- Global parameters

### data_fetchers.py
Functions for retrieving raw data:
- `fetch_mesh_ids()` - Query MeSH IDs for immune system diseases (via SPARQL)
- `fetch_drug_indications()` - Get drug-disease pairs from ChEMBL
- `fetch_drugs()` - Get drug information from ChEMBL
- `fetch_mechanisms_and_targets()` - Get drug mechanisms and protein targets
- `fetch_disease_info()` - Get disease data and targets from Open Targets
- `fetch_clinical_trials()` - Get trial data from ClinicalTrials.gov
- `fetch_reactome_pathways()` - Map UniProt IDs to Reactome pathways

### data_processors.py
Functions for cleaning and transforming data:
- `process_drugs_data()` - Clean drugs, extract properties, add target dictionaries
- `process_diseases_data()` - Process disease information, rename ID columns
- `process_trials_data()` - Flatten nested trial structure, extract objective features
- `process_indications_data()` - Link drug-disease pairs to trial counts (no success classification)
- `add_pathway_annotations()` - Add drug_pathways and disease_pathways columns

### embeddings.py
Functions for semantic analysis:
- `compute_drug_disease_similarities()` - Generate name similarity scores
- `generate_text_embeddings()` - Create embeddings for arbitrary text

### pipeline.py
Main orchestration:
- `run_data_fetching()` - Execute all fetching steps
- `run_data_processing()` - Execute all processing steps
- `run_embeddings()` - Generate embeddings
- `run_full_pipeline()` - Execute complete pipeline

## Data Outputs

### Raw Data (data/raw/)
Directly fetched from APIs, minimal processing

### Processed Data (data/processed/)
Cleaned and structured for analysis:

- **final_drugs_df.pkl**: Drug information with properties, targets, and pathways
- **final_diseases_df.pkl**: Disease information with targets and pathways
- **final_trials_df.pkl**: Clinical trials with flattened features (nct_id, phase, status, enrollment, dates)
- **final_indications_df.pkl**: Drug-disease pairs with trial counts (n_trials, max_trial_phase, has_completed_trial)
- **embeddings_df.pkl**: Semantic similarity scores between drugs and diseases

## Dependencies

Required packages:
- pandas
- numpy
- requests
- lxml (for SPARQL XML parsing)
- chembl_webresource_client
- bioservices
- sentence_transformers
- python-dotenv

## Notes

- All functions support caching - data is only re-fetched when `force=True`
- Raw data is saved to `data/raw/`
- Processed data is saved to `data/processed/`
- The pipeline respects API rate limits (e.g., ClinicalTrials.gov: 10 req/sec)
- UniProt ID mapping can be slow for large datasets
- MeSH IDs are fetched via SPARQL query from the MeSH RDF endpoint (more robust than HTML scraping)
- Disease targets are extracted directly from Open Targets proteinIds (uniprot_swissprot source)
