"""Tests for TLE parsing functions."""

import jax.numpy as jnp
import numpy as np
import pytest

from jaxsgp4 import Satellite, tle2sat, tle2sat_array


# Sample TLE data (Starlink-1008)
TLE_1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
TLE_2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"

# Second satellite (Starlink-1012)
TLE_1B = "1 44718U 19074F   26013.33334491  .00010643  00000+0  66772-3 0  9991"
TLE_2B = "2 44718  53.0643  75.1097 0001446 111.7537 168.3943 15.09820139  5799"


class TestTle2sat:
    def test_returns_satellite(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert isinstance(sat, Satellite)

    def test_mean_motion(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.n0) == pytest.approx(15.10066292, abs=1e-6)

    def test_eccentricity(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.e0) == pytest.approx(0.0002699, abs=1e-7)

    def test_inclination(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.i0) == pytest.approx(53.0657, abs=1e-4)

    def test_argument_of_perigee(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.w0) == pytest.approx(79.3766, abs=1e-4)

    def test_raan(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.Omega0) == pytest.approx(75.1067, abs=1e-4)

    def test_mean_anomaly(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.M0) == pytest.approx(82.4805, abs=1e-4)

    def test_bstar(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.Bstar) == pytest.approx(0.67042e-3, abs=1e-8)

    def test_epoch_year(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert int(sat.epochyr) == 2026

    def test_epoch_days(self):
        sat = tle2sat(TLE_1, TLE_2)
        assert float(sat.epochdays) == pytest.approx(13.33334491, abs=1e-6)

    def test_epoch_year_pre_2000(self):
        """Test that 2-digit years >= 57 map to 1900s."""
        tle_1_old = "1 25544U 98067A   99264.35138889  .00016717  00000+0  10270-3 0  9993"
        tle_2_old = "2 25544  51.5950 320.7367 0005850 130.1587 230.0128 15.50103072  2791"
        sat = tle2sat(tle_1_old, tle_2_old)
        assert int(sat.epochyr) == 1999


class TestTle2satArray:
    def test_returns_satellite(self):
        sat = tle2sat_array([TLE_1, TLE_1B], [TLE_2, TLE_2B])
        assert isinstance(sat, Satellite)

    def test_array_shapes(self):
        sat = tle2sat_array([TLE_1, TLE_1B], [TLE_2, TLE_2B])
        assert sat.n0.shape == (2,)
        assert sat.e0.shape == (2,)
        assert sat.i0.shape == (2,)

    def test_array_values(self):
        sat = tle2sat_array([TLE_1, TLE_1B], [TLE_2, TLE_2B])
        np.testing.assert_allclose(float(sat.n0[0]), 15.10066292, atol=1e-6)
        np.testing.assert_allclose(float(sat.n0[1]), 15.09820139, atol=1e-6)

    def test_single_element_array(self):
        sat = tle2sat_array([TLE_1], [TLE_2])
        assert sat.n0.shape == (1,)
