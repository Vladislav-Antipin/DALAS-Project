# DALAS Drug Repurposing Project

A machine learning project for predicting drug repurposing opportunities using chemical features, clinical data, and graph-based models.

## 🎯 Project Goal

Predict whether approved drugs can be repurposed for new diseases, focusing on autoimmune diseases (psoriasis, IBD, rheumatoid arthritis, etc.).

## 📁 Project Structure

```
DALAS-Project/
├── src/                    # Source code
│   ├── data/              # Data collection and loading
│   ├── features/          # Feature engineering
│   ├── models/            # ML models
│   └── visualization/     # Plotting and dashboard
├── notebooks/             # Jupyter notebooks for EDA
├── tests/                 # Unit tests
├── data/
│   ├── raw/              # Raw data from APIs
│   ├── processed/        # Cleaned and processed data
│   └── cache/            # API response cache
├── results/              # Model outputs, figures
└── pyproject.toml        # Project dependencies
```

## 🚀 Quick Start

### Prerequisites

- Python 3.10 or higher
- [uv](https://github.com/astral-sh/uv) package manager

### Installation

1. **Install uv** (if not already installed):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

2. **Clone the repository**:
```bash
git clone <repository-url>
cd DALAS-Project
```

3. **Create virtual environment and install dependencies**:
```bash
# Create venv and install base dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install project in editable mode with dev dependencies
uv pip install -e ".[dev]"

# Optional: Install graph or visualization dependencies
uv pip install -e ".[graph]"  # For graph models
uv pip install -e ".[viz]"    # For interactive dashboard
```

4. **Set up environment variables**:
```bash
cp .env.example .env
# Edit .env with your API credentials
```

5. **Install pre-commit hooks** (recommended):
```bash
pre-commit install
```

## 🔧 Development Workflow

### Adding Dependencies

```bash
# Add a new package
uv pip install package-name

# Update pyproject.toml manually, then sync
uv pip install -e ".[dev]"
```

### Code Quality

```bash
# Format and lint code
ruff format .
ruff check . --fix

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html
```

### Working with Notebooks

```bash
# Start JupyterLab
jupyter lab

# Notebooks should be saved in notebooks/
```

## 📊 Data Sources

- **ChEMBL API**: Chemical features, molecular descriptors
- **DrugBank**: Drug information (requires academic account)
- **RDKit**: Morgan fingerprints, similarity matrices
- **OpenFDA API**: Approved drug indications (ground truth)
- **Europe PMC API**: Co-citations for drug-disease pairs
- **MeSH**: Disease hierarchy and descriptions

## 🧪 Running the Pipeline

```bash
# 1. Collect data
python -m src.data.collect_drugs
python -m src.data.collect_diseases

# 2. Generate features
python -m src.features.build_features

# 3. Train models
python -m src.models.train

# 4. Evaluate
python -m src.models.evaluate

# 5. Launch dashboard
streamlit run src/visualization/dashboard.py
```

## 🤝 Collaboration Guidelines

### For Collaborators

1. **Always work in a virtual environment** using the uv workflow above
2. **Pull latest changes** before starting work: `git pull`
3. **Run pre-commit hooks** to ensure code quality
4. **Update dependencies**: If you add packages, update `pyproject.toml` and notify your collaborator
5. **Document your notebooks**: Add markdown cells explaining your analysis
6. **Write tests** for any new utility functions in `src/`

### Syncing Environment

When your collaborator adds dependencies:

```bash
git pull
uv pip install -e ".[dev]"
```

## 📝 Testing

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_features.py

# Run with verbose output
pytest -v

# Generate coverage report
pytest --cov=src --cov-report=html
open htmlcov/index.html
```

## 📖 Documentation

See [problem_statement.md](problem_statement.md) for detailed project requirements and methodology.

## 🛠️ Troubleshooting

### RDKit Installation Issues

If RDKit fails to install via pip, use conda:
```bash
conda install -c conda-forge rdkit
```

### API Rate Limits

Some APIs have rate limits. The code implements caching to minimize repeated requests.

## 📜 License

[Add your license here]

## 👥 Contributors

- [Your names here]
