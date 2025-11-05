"""Project configuration and paths."""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Project directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
RAW_DATA_DIR.mkdir(exist_ok=True)
PROCESSED_DATA_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# API credentials
DRUGBANK_USERNAME = os.getenv("DRUGBANK_USERNAME")
DRUGBANK_PASSWORD = os.getenv("DRUGBANK_PASSWORD")

# API endpoints
CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
OPENFDA_API = "https://api.fda.gov/drug"
PUBMED_API = "https://www.ebi.ac.uk/europepmc/webservices/rest"

# Data retrieval endpoints
MESH_URL = "https://www.ncbi.nlm.nih.gov/mesh?Db=mesh&Cmd=DetailsSearch&Term=%22Autoimmune+Diseases%22%5BMeSH+Terms%5D"
OT_URL = "https://api.platform.opentargets.org/api/v4/graphql"
NCT_URL = "https://clinicaltrials.gov/api/v2/studies"

# Model configuration
ST_MODEL = "neuml/pubmedbert-base-embeddings"

# Parameters
NB_TOP_TARGETS = 5  # Number of top targets to retrieve from Open Targets
NB_EVIDENCES = 100  # Number of evidence records to retrieve per disease
