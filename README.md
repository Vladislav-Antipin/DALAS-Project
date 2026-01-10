# DALAS Drug Repurposing Project

Predicting drug repurposing opportunities for autoimmune diseases using machine learning and graph neural networks.

📄 **[Project Report (PDF)](report/aux_main/main.pdf)** · 📋 **[Problem Statement](Docs/problem_statement.md)**

---

## Quick Start

### Prerequisites

Install [uv](https://docs.astral.sh/uv/) (Python package manager):

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### Installation

```bash
git clone https://github.com/Vladislav-Antipin/DALAS-Project.git
cd DALAS-Project
uv sync
```

That's it. `uv sync` creates the virtual environment and installs all dependencies from `pyproject.toml`.

### Running the Project

```bash
# Start Jupyter Lab
uv run jupyter lab

# Run the data pipeline
uv run python -m src.pipeline
```

> **Note:** Using `uv run` automatically uses the project's virtual environment—no manual activation needed.

---

## Project Overview

**Goal:** Predict whether approved drugs can be repurposed for autoimmune diseases (psoriasis, IBD, rheumatoid arthritis, etc.)

**Approach:**

- Collect drug and disease data from ChEMBL, DrugBank, OpenFDA
- Extract molecular features using RDKit (fingerprints, descriptors)
- Train ML models (logistic regression, random forest, gradient boosting)
- Build graph neural networks for drug-disease link prediction

---

## For Collaborators

### First-time Setup

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and install
git clone https://github.com/Vladislav-Antipin/DALAS-Project.git
cd DALAS-Project
uv sync
```

### After Pulling Changes

```bash
git pull
uv sync  # Updates dependencies if pyproject.toml changed
```

### Adding Dependencies

```bash
uv add <package-name>  # Adds to pyproject.toml and installs
```

---

## Project Structure

```text
├── notebooks/          # Jupyter notebooks for analysis
├── src/                # Reusable Python modules
├── data/               # Local data (not committed)
├── report/             # LaTeX report
└── Docs/               # Project documentation
```

---

## Troubleshooting

**RDKit installation issues?**

```bash
# Alternative: use conda for RDKit, then sync other deps
conda install -c conda-forge rdkit
uv sync
```

**Import errors?**

```bash
# Ensure you're using the project environment
uv run python -c "from src.config import DATA_DIR; print(DATA_DIR)"
```

---

## Useful Commands

```bash
uv sync              # Install/update all dependencies
uv add <package>     # Add a new dependency
uv run jupyter lab   # Start Jupyter
uv run python <script>  # Run any Python script
```
