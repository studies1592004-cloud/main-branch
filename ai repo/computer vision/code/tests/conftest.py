"""
tests/conftest.py
==================
Shared pytest configuration and fixtures for the computer vision test suite.

This file is automatically loaded by pytest before any test module runs.
It sets random seeds for reproducibility and provides shared fixtures
that are available to all test files without explicit import.
"""

import numpy as np
import pytest
import random
import os
import sys


# ── Reproducibility ───────────────────────────────────────────────────────────

def pytest_configure(config):
    """Set global random seeds before the test session starts."""
    np.random.seed(42)
    random.seed(42)
    try:
        import torch
        torch.manual_seed(42)
    except ImportError:
        pass


# ── Shared fixtures ───────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def sample_bgr_image():
    """256×256 BGR image shared across the whole test session."""
    rng = np.random.default_rng(0)
    return rng.integers(0, 255, (256, 256, 3), dtype=np.uint8)


@pytest.fixture(scope="session")
def sample_gray_image():
    """256×256 grayscale image shared across the whole test session."""
    rng = np.random.default_rng(1)
    return rng.integers(0, 255, (256, 256), dtype=np.uint8)


@pytest.fixture
def tmp_output_dir(tmp_path):
    """Per-test temporary output directory, cleaned up automatically."""
    d = tmp_path / "outputs"
    d.mkdir()
    return str(d)
