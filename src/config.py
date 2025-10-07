"""Project configuration and paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# API credentials
DRUGBANK_USERNAME = os.getenv("DRUGBANK_USERNAME")
DRUGBANK_PASSWORD = os.getenv("DRUGBANK_PASSWORD")

# API endpoints
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_API = "https://api.fda.gov/drug"
PUBMED_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"
