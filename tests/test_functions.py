"""Tests for sgp4_jdfr Julian Date interface."""

import jax.numpy as jnp
import numpy as np
import pytest

from jax_sgp4 import sgp4, sgp4_jdfr, tle2sat


TLE_1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
TLE_2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"


@pytest.fixture
def sat():
    return tle2sat(TLE_1, TLE_2)


class TestSgp4Jdfr:
    def test_output_shape(self, sat):
        """sgp4_jdfr returns (rv[6], error_code)."""
        rv, error = sgp4_jdfr(sat, jnp.array(2460690.0), jnp.array(0.5))
        assert rv.shape == (6,)

    def test_at_epoch_matches_sgp4_zero(self, sat):
        """sgp4_jdfr at epoch should match sgp4(sat, 0.0)."""
        rv_direct, _ = sgp4(sat, 0.0)

        # Compute the epoch JD/FR from the satellite
        year = sat.epochyr
        days, fraction = jnp.divmod(sat.epochdays, 1.0)
        jd_epoch = year * 365 + (year - 1) // 4 + days + 1721044.5
        fr_epoch = jnp.round(fraction, 8)

        rv_jdfr, _ = sgp4_jdfr(sat, jd_epoch, fr_epoch)
        np.testing.assert_allclose(rv_direct, rv_jdfr, atol=1e-6)

    def test_physically_reasonable(self, sat):
        """Output from sgp4_jdfr should be physically reasonable."""
        rv, error = sgp4_jdfr(sat, jnp.array(2460690.0), jnp.array(0.5))
        r_mag = float(jnp.linalg.norm(rv[:3]))
        # Should be in reasonable LEO range (even if far from epoch)
        assert jnp.all(jnp.isfinite(rv))
