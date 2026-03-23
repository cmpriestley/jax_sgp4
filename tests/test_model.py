"""Tests for the Satellite model."""

from jax_sgp4 import Satellite


def test_satellite_creation():
    """Test that a Satellite NamedTuple can be created with expected fields."""
    sat = Satellite(
        n0=15.1, e0=0.001, i0=53.0, w0=90.0,
        Omega0=180.0, M0=270.0, Bstar=0.0001,
        epochdays=13.33, epochyr=2026,
    )
    assert sat.n0 == 15.1
    assert sat.e0 == 0.001
    assert sat.i0 == 53.0
    assert sat.w0 == 90.0
    assert sat.Omega0 == 180.0
    assert sat.M0 == 270.0
    assert sat.Bstar == 0.0001
    assert sat.epochdays == 13.33
    assert sat.epochyr == 2026


def test_satellite_is_namedtuple():
    """Test Satellite behaves as a NamedTuple (indexable, iterable)."""
    sat = Satellite(
        n0=15.1, e0=0.001, i0=53.0, w0=90.0,
        Omega0=180.0, M0=270.0, Bstar=0.0001,
        epochdays=13.33, epochyr=2026,
    )
    assert sat[0] == 15.1  # n0
    assert len(sat) == 9
