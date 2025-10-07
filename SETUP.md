# Setup Instructions for New Collaborators

## First-Time Setup

### 1. Install uv

**macOS/Linux:**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows:**
```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify installation:
```bash
uv --version
```

### 2. Clone and Setup Project

```bash
# Clone the repository
git clone <repository-url>
cd DALAS-Project

# Option A: Using Makefile (recommended)
make install-dev

# Option B: Manual setup
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
pre-commit install
```

### 3. Configure Environment

```bash
# Copy example environment file
cp .env.example .env

# Edit .env with your credentials
# You'll need:
# - DrugBank username/password (requires academic verification)
# - Other API keys are optional
nano .env  # or use your preferred editor
```

### 4. Verify Setup

```bash
# Activate virtual environment if not already active
source .venv/bin/activate

# Test Python imports
python -c "import numpy, pandas, sklearn; print('✓ Base packages installed')"

# Run example test
pytest tests/ -v
```

## Daily Workflow

### Starting Work

```bash
# 1. Pull latest changes
git pull

# 2. Sync dependencies (if pyproject.toml changed)
make sync
# or: uv pip install -e ".[dev]"

# 3. Activate virtual environment
source .venv/bin/activate
```

### Making Changes

```bash
# Format your code before committing
make format

# Run tests
make test

# Commit (pre-commit hooks will run automatically)
git add .
git commit -m "Your message"
git push
```

### Working with Notebooks

```bash
# Start JupyterLab
make jupyter
# or: jupyter lab

# Save notebooks in the notebooks/ directory
# Clear output before committing: Cell > All Output > Clear
```

## Common Tasks

### Adding a New Package

```bash
# 1. Install the package
uv pip install package-name

# 2. Update pyproject.toml
# Add the package to the appropriate dependencies section

# 3. Notify your collaborator to run:
git pull
make sync
```

### Running Different Components

```bash
# Data collection
python -m src.data.collect_drugs

# Feature engineering
python -m src.features.build_features

# Model training
python -m src.models.train

# Visualization
streamlit run src/visualization/dashboard.py
```

## Troubleshooting

### Virtual Environment Not Activating

```bash
# Remove and recreate
rm -rf .venv
uv venv
source .venv/bin/activate
uv pip install -e ".[dev]"
```

### RDKit Installation Fails

RDKit can be tricky with pip. Try:

```bash
# Option 1: Use conda for RDKit only
conda install -c conda-forge rdkit
uv pip install -e ".[dev]"

# Option 2: Use mamba (faster than conda)
mamba install -c conda-forge rdkit
uv pip install -e ".[dev]"
```

### Pre-commit Hooks Failing

```bash
# Update hooks
pre-commit autoupdate

# Run manually to see issues
pre-commit run --all-files

# Skip hooks if needed (not recommended)
git commit --no-verify
```

### Dependency Conflicts

```bash
# Check what's installed
uv pip list

# Force reinstall from pyproject.toml
rm -rf .venv
make install-dev
```

## Project Structure Reminder

```
src/
├── data/          # Put data collection scripts here
├── features/      # Put feature engineering code here
├── models/        # Put model training/evaluation code here
└── visualization/ # Put plotting and dashboard code here

notebooks/         # EDA and experiments (keep organized!)
tests/            # Unit tests (mirror src/ structure)
data/
├── raw/          # Original data from APIs
├── processed/    # Cleaned data
└── cache/        # Cached API responses
```

## Best Practices

1. **Always use the virtual environment** - Check with `which python`
2. **Run tests before pushing** - `make test`
3. **Format code** - `make format` or rely on pre-commit hooks
4. **Clear notebook outputs** before committing
5. **Cache API responses** to avoid rate limits
6. **Don't commit large files** (data, models) - use `.gitignore`
7. **Document your code** with docstrings and comments
8. **Update README** if you add new features or change workflow

## Getting Help

- Check [README.md](README.md) for general project info
- See [problem_statement.md](problem_statement.md) for project goals
- Run `make help` to see available make commands
- Contact your collaborator if stuck!
