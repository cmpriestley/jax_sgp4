"""Tests for package version."""

import jaxsgp4


def test_version_exists():
    assert hasattr(jaxsgp4, "__version__")


def test_version_format():
    parts = jaxsgp4.__version__.split(".")
    assert len(parts) == 3
    for part in parts:
        assert part.isdigit()
