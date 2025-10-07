"""Configuration management for the DALAS project."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / os.getenv("DATA_DIR", "data")
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"
RESULTS_DIR = PROJECT_ROOT / os.getenv("RESULTS_DIR", "results")

# Create directories if they don't exist
for directory in [RAW_DATA_DIR, PROCESSED_DATA_DIR, CACHE_DIR, RESULTS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# API Configuration
DRUGBANK_USERNAME = os.getenv("DRUGBANK_USERNAME")
DRUGBANK_PASSWORD = os.getenv("DRUGBANK_PASSWORD")
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY")

# API endpoints
CHEMBL_BASE_URL = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_BASE_URL = "https://api.fda.gov/drug"
EUROPE_PMC_BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Model parameters (can be overridden)
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
