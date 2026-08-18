import shutil
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_COMMON = REPO_ROOT / "examples" / "research-crew" / "common"


@pytest.fixture()
def example_common_dir() -> Path:
    """Path to the shipped research-crew example's common/ folder."""
    return EXAMPLE_COMMON


@pytest.fixture()
def tmp_project(tmp_path: Path):
    """A fresh, mutable copy of the example project for tests that mutate it."""
    dest = tmp_path / "common"
    shutil.copytree(EXAMPLE_COMMON, dest)
    return dest
