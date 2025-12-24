# DALAS Drug Repurposing Project

Predicting drug repurposing opportunities for immune system diseases using machine learning.

[Project Report (PDF)](report/aux_main/main.pdf)

# UV package manager

### 1. Install uv (if you don't have it)

```bash
# macOS/Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

# Verify it works
uv --version
```

### 2. Set up the project

```bash
# Clone and navigate to the project
cd DALAS-Project

# Create virtual environment
uv venv

# Activate it
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install all dependencies
uv pip install -e .
```

### 3. Configure API credentials

```bash
# Copy the example file
cp .env.example .env

# Edit with your credentials
nano .env  # or use any editor
```

### 4. Start working!

```bash
# Option A: Start Jupyter for analysis
jupyter lab

# Option B: Run Python scripts
python -m src.your_script

# Option C: Use Python interactively
python
>>> from src.config import DATA_DIR
>>> print(DATA_DIR)
```

## 📊 What We're Building

See [problem_statement.md](problem_statement.md) for the full project plan.

**Quick summary:**
- Collect drug data from ChEMBL, DrugBank, OpenFDA
- Extract chemical features (RDKit fingerprints, descriptors)
- Build ML models (logistic regression, random forest, gradient boosting)
- Optional: Graph neural networks for drug-disease prediction
- Create dashboard to explore results

## 🤝 For Collaborators

When someone new joins:

```bash
# 1. Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Clone and setup
git clone <repo-url>
cd DALAS-Project
uv venv
source .venv/bin/activate
uv pip install -e .

# 3. Add your API credentials
cp .env.example .env
nano .env
```

**When pulling new changes:**
```bash
git pull
uv pip install -e .  # Updates dependencies if pyproject.toml changed
```

## 💡 Tips

- **Keep notebooks organized**: Name them with numbers (e.g., `01_data_collection.ipynb`)
- **Put reusable code in `src/`**: Import it in notebooks with `from src.config import ...`
- **Don't commit large files**: Data and models stay local (already in `.gitignore`)
- **Share your work**: Push code and notebooks, but data stays on your machine

## 🆘 Common Issues

**RDKit won't install?**
```bash
# Use conda instead
conda install -c conda-forge rdkit
uv pip install -e .
```

**Import errors?**
```bash
# Make sure virtual environment is activated
source .venv/bin/activate

# Check you're using the right Python
which python  # Should show .venv/bin/python
```

## 📚 Useful Commands

```bash
# Start Jupyter
jupyter lab

# Check what's installed
uv pip list

# Add a new package
uv pip install package-name
# Then add it to pyproject.toml manually

# Deactivate virtual environment
deactivate
```

---

**Questions?** Check the [problem statement](problem_statement.md) or ask your colleague!
