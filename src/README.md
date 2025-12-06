# DALAS Data Pipeline

This directory contains modular Python scripts for fetching, processing, and preparing drug-disease data for machine learning-based drug repurposing prediction.

## Structure

```
src/
├── config.py              # Configuration, paths, and constants
├── data_fetchers.py       # Functions to fetch raw data from APIs
├── data_processors.py     # Functions to clean and process raw data
├── embeddings.py          # Functions to generate name similarity embeddings
├── feature_engineering.py # Fingerprints, pathway TF-IDF, label creation
├── data_merger.py         # Merge all features into ML-ready dataset
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
   ├─ Process Drugs     → data/processed/final_drugs_df.pkl
   ├─ Process Diseases  → data/processed/final_diseases_df.pkl
   ├─ Process Trials    → data/processed/final_trials_df.pkl
   └─ Process Indications → data/processed/final_indications_df.pkl

3. EMBEDDINGS (embeddings.py)
   └─ Drug-Disease Similarities → data/processed/embeddings_df.pkl

4. FEATURE ENGINEERING (feature_engineering.py)
   ├─ Fingerprints      → data/processed/fingerprints_df.pkl
   ├─ Pathway TF-IDF    → data/processed/drugs_with_pathways.pkl
   │                    → data/processed/diseases_with_pathways.pkl
   └─ Labels            → data/processed/labeled_indications_df.pkl

5. DATA MERGING (data_merger.py)
   └─ ML-Ready Dataset  → data/processed/merged_df.pkl
                        → results/merged_df.pkl (copy)

6. MODEL TRAINING (model_training.py)
   ├─ RandomForest      → results/random_forest_model.pkl
   ├─ LogisticRegression → results/logistic_regression_model.pkl
   ├─ Feature Importance → results/rf_feature_importance.csv
   │                     → results/lr_feature_importance.csv
   └─ Metrics           → results/model_metrics.pkl
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

# Only run feature engineering (requires processed data)
python -m src.pipeline --step features

# Only merge into final dataset (requires all previous steps)
python -m src.pipeline --step merge

# Only train models (requires merged dataset)
python -m src.pipeline --step train

# Force recompute a specific step
python -m src.pipeline --step features --force
python -m src.pipeline --step train --force
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

### feature_engineering.py
Feature creation for ML:
- `compute_fingerprints()` - RDKit fingerprints → PCA (100 components)
- `compute_pathway_features()` - TF-IDF on pathways → SVD (50 components)
- `engineer_labels()` - Create binary labels from phase information

### data_merger.py
Final dataset preparation:
- `merge_all_features()` - Combine all features into ML-ready dataset
- `get_feature_summary()` - Generate feature statistics

### model_training.py
ML model training:
- `run_model_training()` - Train RandomForest and LogisticRegression
- `predict_new_pairs()` - Predict on new drug-disease pairs
- `get_feature_importance()` - Extract feature importances from models

### pipeline.py
Main orchestration:
- `run_data_fetching()` - Execute all fetching steps
- `run_data_processing()` - Execute all processing steps
- `run_embeddings()` - Generate embeddings
- `run_features()` - Execute feature engineering
- `run_merge()` - Create final ML dataset
- `run_training()` - Train ML models
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
- **fingerprints_df.pkl**: PCA-reduced molecular fingerprints (100 components)
- **drugs_with_pathways.pkl**: Drugs with TF-IDF pathway features (50 components)
- **diseases_with_pathways.pkl**: Diseases with TF-IDF pathway features (50 components)
- **labeled_indications_df.pkl**: Indications with binary labels
- **merged_df.pkl**: Final ML-ready dataset with all features

## Design Decisions

### Label Engineering

**How we define success/failure:**

| Label | Criteria | Rationale |
|-------|----------|-----------|
| **True** | `max_phase_for_ind == 4` | Phase 4 = approved drug, objective evidence of efficacy |
| **False** | `max_phase_for_ind < 2` | Phase 0-1, no efficacy signal in humans yet |
| **Excluded** | `max_phase_for_ind in [2, 3]` | Outcome uncertain, trials may be ongoing |

**Why NOT use p-values from trials:**
- P-values are missing in ~70% of trials
- Interpreting "success" from p-values is subjective (threshold choice)
- Multiple endpoints make aggregation ambiguous
- This was in the original notebook but adds label noise

### Fingerprints

- **Algorithm**: RDKit fingerprints (2048-bit), more interpretable than Morgan/ECFP
- **Reduction**: PCA to 100 components (retains ~95% variance)
- **Handling**: Drugs without valid SMILES are excluded (typically biologics)

### Pathway Features

- **Method**: TF-IDF vectorization of Reactome pathway IDs
- **Purpose**: Downweight ubiquitous "housekeeping" pathways, highlight specific ones
- **Reduction**: SVD to 50 components
- **Corpus**: Combined drug + disease pathways for consistent feature space

### Data Filtering

- **Biotherapeutics excluded**: Different mechanisms than small molecules
- **Missing labels excluded**: Unknown outcomes add noise
- **High-missing columns dropped**: >50% missing values

### Model Training

- **Models**: RandomForest (300 trees) and LogisticRegression
- **Class weighting**: `balanced` to handle imbalanced classes
- **Preprocessing**: Mean imputation + StandardScaler
- **Split**: 80/20 stratified train/test
- **Evaluation**: Accuracy, Precision, Recall, F1, AUC-ROC
- **Primary metric**: AUC-ROC (handles class imbalance well)

## Dependencies

Required packages:
- pandas, numpy
- requests, lxml (for SPARQL XML parsing)
- chembl_webresource_client, bioservices
- sentence_transformers
- scikit-learn (for PCA, TF-IDF, SVD)
- rdkit (for fingerprints) - install via conda

## Notes

- All functions support caching - data is only re-fetched when `force=True`
- Raw data is saved to `data/raw/`
- Processed data is saved to `data/processed/`
- The pipeline respects API rate limits (e.g., ClinicalTrials.gov: 10 req/sec)
- UniProt ID mapping can be slow for large datasets
- MeSH IDs are fetched via SPARQL query from the MeSH RDF endpoint
- Disease targets are extracted directly from Open Targets proteinIds

## Comparison with Notebooks

| Step | Notebook | Script | Notes |
|------|----------|--------|-------|
| Data fetching | `01_1` | `data_fetchers.py` | Equivalent |
| Fingerprints | `01_2` | `feature_engineering.py` | Equivalent (RDKit → PCA) |
| Data processing | `02` | `data_processors.py` | Scripts use objective trial counts, not p-values |
| Pathway TF-IDF | `02` | `feature_engineering.py` | Equivalent (TF-IDF → SVD) |
| Labels | `02` | `feature_engineering.py` | Scripts use Phase 4 only (cleaner) |
| Merge | `03` | `data_merger.py` | Equivalent filtering |
| Embeddings | `01_1` | `embeddings.py` | Scripts compute on processed data (more efficient) |
