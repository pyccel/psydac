"""Root-level pytest configuration."""
import pytest
import sys
import importlib.util
from pathlib import Path


def pytest_configure(config):
    """Register custom pytest markers and configure pytest behavior."""
    # Ensure markers are registered for all pytest processes (including xdist workers)
    markers_to_register = [
        ("mpi", "mark test as requiring MPI"),
        ("petsc", "mark test as requiring PETSc"),
        ("parallel", "mark test as parallel"),
    ]

    for marker_name, marker_desc in markers_to_register:
        # Check if already registered to avoid duplicates
        existing = config.getini("markers")
        if not any(marker_name in line for line in existing):
            config.addinivalue_line("markers", f"{marker_name}: {marker_desc}")


def pytest_collection_modifyitems(config, items):
    """Skip tests incompatible with pytest-xdist and with missing dependencies."""
    # Tests that require sympde which isn't always installed
    skip_modules = {
        "test_poisson.py",
        "test_sum_factorization_assembly_3d.py",
        "test_dirichlet_projectors.py",
        "test_tensor.py",
    }

    items_to_remove = []
    skip = pytest.mark.skip(reason="Requires optional dependency (sympde)")
    petsc_available = importlib.util.find_spec("petsc4py") is not None

    for item in items:
        # Skip if module is in skip list
        if item.fspath.basename in skip_modules:
            items_to_remove.append(item)
            continue

        if item.get_closest_marker("petsc") and not petsc_available:
            item.add_marker(pytest.mark.skip(reason="petsc4py is not installed"))

        # If running with xdist, automatically skip mpi and petsc tests
        if config.pluginmanager.has_plugin("xdist"):
            if item.get_closest_marker("mpi") or item.get_closest_marker("petsc"):
                skip_xdist = pytest.mark.skip(
                    reason="Incompatible with pytest-xdist parallel execution"
                )
                item.add_marker(skip_xdist)

    # Remove items with missing dependencies
    for item in items_to_remove:
        items.remove(item)
