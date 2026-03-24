"""Tests for JAX transformation compatibility (jit, vmap, grad)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from jaxsgp4 import Satellite, sgp4, sgp4_jdfr, tle2sat, tle2sat_array


# Sample TLEs
TLE_1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
TLE_2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"
TLE_1B = "1 44718U 19074F   26013.33334491  .00010643  00000+0  66772-3 0  9991"
TLE_2B = "2 44718  53.0643  75.1097 0001446 111.7537 168.3943 15.09820139  5799"


@pytest.fixture
def sat():
    return tle2sat(TLE_1, TLE_2)


@pytest.fixture
def sat_array():
    return tle2sat_array([TLE_1, TLE_1B], [TLE_2, TLE_2B])


class TestJit:
    def test_jit_sgp4(self, sat):
        """sgp4 can be JIT-compiled."""
        jitted_sgp4 = jax.jit(sgp4)
        rv, error = jitted_sgp4(sat, 0.0)
        assert rv.shape == (6,)
        assert int(error) == 0

    def test_jit_produces_same_result(self, sat):
        """JIT result matches eager result."""
        rv_eager, _ = sgp4(sat, 60.0)
        rv_jit, _ = jax.jit(sgp4)(sat, 60.0)
        np.testing.assert_allclose(rv_eager, rv_jit, rtol=1e-4, atol=1e-2)

    def test_jit_sgp4_jdfr(self, sat):
        """sgp4_jdfr can be JIT-compiled."""
        jitted = jax.jit(sgp4_jdfr)
        rv = jitted(sat, jnp.array(2460690.0), jnp.array(0.5))
        # sgp4_jdfr returns the full output from sgp4 (rv_array, error_code)
        assert rv[0].shape == (6,)


class TestVmap:
    def test_vmap_over_times(self, sat):
        """Vectorize sgp4 over multiple time points."""
        times = jnp.array([0.0, 30.0, 60.0, 90.0])
        vmapped = jax.vmap(sgp4, in_axes=(None, 0))
        rvs, errors = vmapped(sat, times)
        assert rvs.shape == (4, 6)
        assert errors.shape == (4,)

    def test_vmap_over_satellites(self, sat_array):
        """Vectorize sgp4 over multiple satellites."""
        vmapped = jax.vmap(sgp4, in_axes=(0, None))
        rvs, errors = vmapped(sat_array, 0.0)
        assert rvs.shape == (2, 6)
        assert errors.shape == (2,)

    def test_vmap_satellites_and_times(self, sat_array):
        """Nested vmap: multiple satellites x multiple times."""
        times = jnp.array([0.0, 30.0, 60.0])
        # Inner vmap: over times for a single satellite
        vmap_times = jax.vmap(sgp4, in_axes=(None, 0))
        # Outer vmap: over satellites
        vmap_sats_times = jax.vmap(vmap_times, in_axes=(0, None))
        rvs, errors = vmap_sats_times(sat_array, times)
        assert rvs.shape == (2, 3, 6)  # 2 sats x 3 times x 6 state
        assert errors.shape == (2, 3)

    def test_jit_vmap_combined(self, sat_array):
        """JIT + vmap combined."""
        times = jnp.array([0.0, 30.0, 60.0])
        vmap_times = jax.vmap(sgp4, in_axes=(None, 0))
        vmap_sats_times = jax.jit(jax.vmap(vmap_times, in_axes=(0, None)))
        rvs, errors = vmap_sats_times(sat_array, times)
        assert rvs.shape == (2, 3, 6)


class TestGrad:
    def test_grad_wrt_tsince(self, sat):
        """Gradient of position norm w.r.t. propagation time."""
        def pos_norm(tsince):
            rv, _ = sgp4(sat, tsince)
            return jnp.linalg.norm(rv[:3])

        grad_fn = jax.grad(pos_norm)
        g = grad_fn(0.0)
        assert jnp.isfinite(g)

    def test_grad_wrt_orbital_element(self):
        """Gradient of position w.r.t. an orbital element (inclination)."""
        def pos_x(i0):
            sat = Satellite(
                n0=jnp.array(15.1), e0=jnp.array(0.001), i0=i0,
                w0=jnp.array(90.0), Omega0=jnp.array(180.0),
                M0=jnp.array(270.0), Bstar=jnp.array(0.0001),
                epochdays=jnp.array(13.33), epochyr=jnp.array(2026.0),
            )
            rv, _ = sgp4(sat, 60.0)
            return rv[0]  # x-position

        grad_fn = jax.grad(pos_x)
        g = grad_fn(jnp.array(53.0))
        assert jnp.isfinite(g)
        assert float(g) != 0.0  # Inclination should affect x-position

    def test_jacobian(self, sat):
        """Full Jacobian of state vector w.r.t. time."""
        def propagate(tsince):
            rv, _ = sgp4(sat, tsince)
            return rv

        jac = jax.jacobian(propagate)(60.0)
        assert jac.shape == (6,)
        assert jnp.all(jnp.isfinite(jac))
