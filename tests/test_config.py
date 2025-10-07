"""Tests for configuration module."""

from pathlib import Path

from src.config import (
    CACHE_DIR,
    DATA_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    RAW_DATA_DIR,
    RESULTS_DIR,
)


def test_project_root_exists():
    """Test that project root is correctly identified."""
    assert PROJECT_ROOT.exists()
    assert (PROJECT_ROOT / "pyproject.toml").exists()


def test_directories_exist():
    """Test that required directories are created."""
    assert DATA_DIR.exists()
    assert RAW_DATA_DIR.exists()
    assert PROCESSED_DATA_DIR.exists()
    assert CACHE_DIR.exists()
    assert RESULTS_DIR.exists()


def test_directories_are_paths():
    """Test that directory variables are Path objects."""
    assert isinstance(DATA_DIR, Path)
    assert isinstance(RAW_DATA_DIR, Path)
    assert isinstance(PROCESSED_DATA_DIR, Path)
    assert isinstance(CACHE_DIR, Path)
    assert isinstance(RESULTS_DIR, Path)
