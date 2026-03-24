"""TLE (Two-Line Element) parsing utilities."""

from .model import Satellite
import jax.numpy as jnp
import numpy as np

def tle2sat(tle_1, tle_2):
    """Extract orbital elements from a TLE and return a Satellite object.

    Inputs:
      tle_1 : str : First line of TLE
      tle_2 : str : Second line of TLE

    Returns:
      Satellite : NamedTuple containing orbital elements as JAX scalars
    """
    n0 = jnp.array(float(tle_2[52:63]))                             # Mean motion (revs/day)
    e0 = jnp.array(float('0.' + tle_2[26:33].replace(' ', '0')))    # Eccentricity
    i0 = jnp.array(float(tle_2[8:16]))                              # Inclination (degrees)
    w0 = jnp.array(float(tle_2[34:42]))                             # Argument of perigee (degrees)
    Omega0 = jnp.array(float(tle_2[17:25]))                         # RAAN (degrees)
    M0 = jnp.array(float(tle_2[43:51]))                             # Mean anomaly (degrees)
    epochdays = jnp.array(float(tle_1[20:32]))                      # Epoch as fractional day of year

    # Bstar: mantissa with implicit decimal + exponent
    Bstar = jnp.array(float(tle_1[53] + '.' + tle_1[54:59]))  
    bexp = int(tle_1[59:61])  
    Bstar = Bstar * 10 ** bexp                                      # Drag coefficient (Earth radii^-1) 

    # Epoch year: 2-digit to 4-digit conversion (57 threshold per NORAD convention)
    two_digit_year = int(tle_1[18:20])
    epochyr = 2000 + two_digit_year if two_digit_year < 57 else 1900 + two_digit_year

    return Satellite(n0, e0, i0, w0, Omega0, M0, Bstar, epochdays, epochyr)


def tle2sat_array(tle_1_array, tle_2_array):  
    """Extract orbital elements from a TLE and return a Satellite object.

    Inputs:
      tle_1 : str : First line of TLE
      tle_2 : str : Second line of TLE

    Returns:
      Satellite : NamedTuple containing orbital elements as JAX scalars
    """ 

    n = len(tle_1_array)                                                      

    # Pre-allocate NumPy arrays
    n0 = np.empty(n)
    e0 = np.empty(n)
    i0 = np.empty(n)
    w0 = np.empty(n)
    Omega0 = np.empty(n)
    M0 = np.empty(n)
    Bstar = np.empty(n)
    epochdays = np.empty(n)
    epochyr = np.empty(n)

    for idx in range(n):
        tle_1 = tle_1_array[idx]
        tle_2 = tle_2_array[idx]

        n0[idx] = float(tle_2[52:63])
        e0[idx] = float('0.' + tle_2[26:33].replace(' ', '0'))
        i0[idx] = float(tle_2[8:16])
        w0[idx] = float(tle_2[34:42])
        Omega0[idx] = float(tle_2[17:25])
        M0[idx] = float(tle_2[43:51])
        epochdays[idx] = float(tle_1[20:32])

        bstar_mantissa = float(tle_1[53] + '.' + tle_1[54:59])
        bexp = int(tle_1[59:61])
        Bstar[idx] = bstar_mantissa * 10 ** bexp

        two_digit_year = int(tle_1[18:20])
        epochyr[idx] = 2000 + two_digit_year if two_digit_year < 57 else 1900 + two_digit_year

    # Single bulk conversion to JAX
    return Satellite(
        n0=jnp.array(n0),
        e0=jnp.array(e0),
        i0=jnp.array(i0),
        w0=jnp.array(w0),
        Omega0=jnp.array(Omega0),
        M0=jnp.array(M0),
        Bstar=jnp.array(Bstar),
        epochdays=jnp.array(epochdays),
        epochyr=jnp.array(epochyr),
    )