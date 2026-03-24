"""Tests for SGP4 propagation accuracy and correctness."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsgp4 import Satellite, sgp4, tle2sat


# Sample TLE (Starlink-1008)
TLE_1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
TLE_2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"


@pytest.fixture
def sat():
    return tle2sat(TLE_1, TLE_2)


class TestSgp4Output:
    def test_output_shape(self, sat):
        """sgp4 returns (rv_array[6], error_code)."""
        rv, error_code = sgp4(sat, 0.0)
        assert rv.shape == (6,)

    def test_position_velocity_split(self, sat):
        """First 3 elements are position (km), last 3 are velocity (km/s)."""
        rv, _ = sgp4(sat, 0.0)
        r = rv[:3]
        v = rv[3:]
        assert r.shape == (3,)
        assert v.shape == (3,)

    def test_zero_tsince(self, sat):
        """At epoch (tsince=0), output should be physically reasonable."""
        rv, error_code = sgp4(sat, 0.0)
        r = rv[:3]
        v = rv[3:]

        # Position magnitude: LEO orbit ~6500-7000 km from Earth center
        r_mag = jnp.linalg.norm(r)
        assert 6000.0 < float(r_mag) < 7200.0

        # Velocity magnitude: LEO ~7-8 km/s
        v_mag = jnp.linalg.norm(v)
        assert 6.5 < float(v_mag) < 8.5

        # No error
        assert int(error_code) == 0

    def test_propagation_forward(self, sat):
        """Propagation at different times gives different results."""
        rv0, _ = sgp4(sat, 0.0)
        rv1, _ = sgp4(sat, 60.0)  # 1 hour later

        # Positions should differ
        assert not jnp.allclose(rv0[:3], rv1[:3])

    def test_orbit_altitude_reasonable(self, sat):
        """Check altitude stays reasonable over one orbit (~95 min for Starlink)."""
        earth_radius = 6378.135
        for t in [0.0, 20.0, 47.5, 70.0, 95.0]:
            rv, error = sgp4(sat, t)
            r_mag = float(jnp.linalg.norm(rv[:3]))
            altitude = r_mag - earth_radius
            # Starlink orbits ~550 km altitude
            assert 400.0 < altitude < 700.0, f"Altitude {altitude} km at t={t} min"
            assert int(error) == 0


class TestSgp4ErrorCodes:
    def test_no_error_normal_orbit(self, sat):
        _, error_code = sgp4(sat, 0.0)
        assert int(error_code) == 0

    def test_error_code_is_scalar(self, sat):
        _, error_code = sgp4(sat, 0.0)
        assert error_code.shape == ()


class TestSgp4NumericalStability:
    def test_small_tsince(self, sat):
        """Very small propagation time should not cause numerical issues."""
        rv, error = sgp4(sat, 0.001)
        assert int(error) == 0
        assert jnp.all(jnp.isfinite(rv))

    def test_large_tsince(self, sat):
        """Propagation over several days (TLE accuracy degrades but should not crash)."""
        rv, error = sgp4(sat, 1440.0 * 7)  # 7 days
        assert jnp.all(jnp.isfinite(rv))

    def test_negative_tsince(self, sat):
        """Backward propagation should work."""
        rv, error = sgp4(sat, -60.0)
        assert jnp.all(jnp.isfinite(rv))
        r_mag = float(jnp.linalg.norm(rv[:3]))
        assert 6000.0 < r_mag < 7200.0
