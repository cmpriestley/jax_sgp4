"""Tests for package version."""

import jax_sgp4


def test_version_exists():
    assert hasattr(jax_sgp4, "__version__")


def test_version_format():
    parts = jax_sgp4.__version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()
