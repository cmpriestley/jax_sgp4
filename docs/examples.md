# Usage Examples

This guide demonstrates how to use `jax_sgp4` for satellite orbit propagation, including how to leverage JAX transformations for performance and differentiability.

## Basic Propagation

Parse a TLE and propagate to a time offset from epoch:

```python
from jax_sgp4 import tle2sat, sgp4

# Two-Line Element set for a satellite
tle_line1 = "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990"
tle_line2 = "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798"

sat = tle2sat(tle_line1, tle_line2)

# Propagate 60 minutes from epoch
rv, error_code = sgp4(sat, 60.0)

r = rv[:3]   # Position [x, y, z] in km (TEME frame)
v = rv[3:]   # Velocity [vx, vy, vz] in km/s (TEME frame)

print(f"Position: {r} km")
print(f"Velocity: {v} km/s")
print(f"Error code: {error_code}")  # 0 = no error
```

## Using Julian Dates

If you prefer to specify the target time as a Julian Date rather than minutes-since-epoch:

```python
import jax.numpy as jnp
from jax_sgp4 import tle2sat, sgp4_jdfr

sat = tle2sat(tle_line1, tle_line2)

# Propagate to a specific Julian Date
jd = jnp.array(2460690.0)   # Integer part of Julian Date
fr = jnp.array(0.5)         # Fractional day

rv, error_code = sgp4_jdfr(sat, jd, fr)
```

## JIT Compilation

Wrap the propagator with `jax.jit` to compile it into an optimised kernel. This eliminates Python overhead and is essential for repeated evaluations:

```python
import jax

jitted_sgp4 = jax.jit(sgp4)

# First call triggers compilation (slower)
rv, error = jitted_sgp4(sat, 0.0)

# Subsequent calls are fast (~microseconds)
rv, error = jitted_sgp4(sat, 60.0)
```

## Vectorizing Over Time Steps

Use `jax.vmap` to propagate a single satellite to many time points in parallel:

```python
import jax
import jax.numpy as jnp
from jax_sgp4 import tle2sat, sgp4

sat = tle2sat(tle_line1, tle_line2)

# Create an array of time steps (0 to 1440 minutes = 1 day, every minute)
times = jnp.arange(0, 1440, 1.0)

# Vectorize over the time argument (index 1), keep satellite fixed (None)
sgp4_many_times = jax.vmap(sgp4, in_axes=(None, 0))

# Propagate to all times at once
rvs, errors = sgp4_many_times(sat, times)

print(f"Output shape: {rvs.shape}")  # (1440, 6)
print(f"Error codes shape: {errors.shape}")  # (1440,)

# Extract positions and velocities
positions = rvs[:, :3]   # (1440, 3) array of positions in km
velocities = rvs[:, 3:]  # (1440, 3) array of velocities in km/s
```

## Vectorizing Over Satellites

Parse multiple TLEs and propagate all satellites to a single time:

```python
import jax
from jax_sgp4 import tle2sat_array, sgp4

# Multiple TLE lines
tle_1_lines = [
    "1 44714U 19074B   26013.33334491  .00010762  00000+0  67042-3 0  9990",
    "1 44718U 19074F   26013.33334491  .00010643  00000+0  66772-3 0  9991",
]
tle_2_lines = [
    "2 44714  53.0657  75.1067 0002699  79.3766  82.4805 15.10066292  5798",
    "2 44718  53.0643  75.1097 0001446 111.7537 168.3943 15.09820139  5799",
]

# Parse into a vectorized Satellite object (each field is an array)
sats = tle2sat_array(tle_1_lines, tle_2_lines)

# Vectorize over satellites (index 0), keep time fixed (None)
sgp4_many_sats = jax.vmap(sgp4, in_axes=(0, None))

rvs, errors = sgp4_many_sats(sats, 60.0)

print(f"Output shape: {rvs.shape}")  # (2, 6)
```

## Vectorizing Over Both Satellites and Times

Compose two `vmap` calls to propagate N satellites to M time steps, producing an (N, M, 6) output:

```python
import jax
import jax.numpy as jnp
from jax_sgp4 import tle2sat_array, sgp4

# Parse satellites
sats = tle2sat_array(tle_1_lines, tle_2_lines)

# Time steps
times = jnp.arange(0, 1440, 10.0)  # Every 10 minutes for a day

# Inner vmap: vectorize a single satellite over many times
sgp4_over_times = jax.vmap(sgp4, in_axes=(None, 0))

# Outer vmap: vectorize over satellites
sgp4_parallel = jax.vmap(sgp4_over_times, in_axes=(0, None))

# Optional: JIT-compile the whole thing for maximum performance
sgp4_parallel = jax.jit(sgp4_parallel)

rvs, errors = sgp4_parallel(sats, times)

print(f"Output shape: {rvs.shape}")    # (2, 144, 6) = satellites x times x state
print(f"Errors shape: {errors.shape}")  # (2, 144)
```

## Loading TLEs from a File

The package includes sample Starlink TLE data. Here's how to load TLEs from a text file:

```python
from jax_sgp4 import tle2sat_array

# Read a 3-line TLE file (name, line1, line2 repeating)
with open("path/to/tles.txt") as f:
    lines = [line.strip() for line in f if line.strip()]

# Extract line 1 and line 2 entries (skip name lines)
tle_1_lines = [lines[i] for i in range(1, len(lines), 3)]
tle_2_lines = [lines[i] for i in range(2, len(lines), 3)]

sats = tle2sat_array(tle_1_lines, tle_2_lines)
```

## Computing Gradients

Since `jax_sgp4` is a pure JAX implementation, you can compute exact gradients using automatic differentiation. This is useful for orbit determination and optimisation.

### Gradient with Respect to Time

```python
import jax
import jax.numpy as jnp
from jax_sgp4 import tle2sat, sgp4

sat = tle2sat(tle_line1, tle_line2)

# Define a scalar function of time (e.g., orbital radius)
def orbital_radius(tsince):
    rv, _ = sgp4(sat, tsince)
    return jnp.linalg.norm(rv[:3])

# Compute the derivative of radius with respect to time
d_radius_dt = jax.grad(orbital_radius)
print(f"dr/dt at t=0: {d_radius_dt(0.0)} km/min")
```

### Gradient with Respect to Orbital Elements

```python
import jax
import jax.numpy as jnp
from jax_sgp4 import Satellite, sgp4

def position_x(inclination):
    """X-position as a function of inclination."""
    sat = Satellite(
        n0=jnp.array(15.1), e0=jnp.array(0.001), i0=inclination,
        w0=jnp.array(90.0), Omega0=jnp.array(180.0),
        M0=jnp.array(270.0), Bstar=jnp.array(0.0001),
        epochdays=jnp.array(13.33), epochyr=jnp.array(2026.0),
    )
    rv, _ = sgp4(sat, 60.0)
    return rv[0]

grad_fn = jax.grad(position_x)
print(f"dx/di: {grad_fn(jnp.array(53.0))}")
```

### Full Jacobian of the State Vector

```python
import jax
from jax_sgp4 import tle2sat, sgp4

sat = tle2sat(tle_line1, tle_line2)

def propagate(tsince):
    rv, _ = sgp4(sat, tsince)
    return rv  # [x, y, z, vx, vy, vz]

# 6x1 Jacobian: d(state)/d(time)
jac = jax.jacobian(propagate)(60.0)
print(f"Jacobian shape: {jac.shape}")  # (6,)
```

## GPU Acceleration

JAX automatically uses GPU hardware when available. No code changes are needed — just ensure `jaxlib` is installed with GPU support:

```bash
pip install --upgrade "jax[cuda12]"
```

The propagator will then run on GPU, which provides large speedups when vectorizing over many satellites or time steps.

## Precision

By default, JAX uses 32-bit floating point. To enable 64-bit precision:

```python
import jax
jax.config.update("jax_enable_x64", True)
```

Set this **before** any JAX operations. 64-bit precision gives results matching reference C++ implementations to ~10⁻¹¹ km, but 32-bit is generally sufficient for operational use given the inherent accuracy limits of TLE data.
