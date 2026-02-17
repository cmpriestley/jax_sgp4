
from .model import Satellite
import jax.numpy as jnp
import numpy as np

def tle2sat(tle_1, tle_2):

    """
    Extract orbital elements from TLE (Two-Line Element) data and store them in a Satellite object.

    Inputs:
      tle_1 : str : First line of TLE
      tle_2 : str : Second line of TLE

    Returns:
      Satellite : NamedTuple containing orbital elements
    """
    # could add checks for valid TLE format later (as in original code)
    # there are also checks for non ascii characters in original code
    # just do necessary element extraction for now (could add e.g. satnum etc. later)

    # Extract elements from TLE strings
    # Use jnp.array() instead of jnp.asarray() to ensure JAX arrays are created
    # (jnp.asarray may return Python floats in some environments)
    n0 = jnp.array(float(tle_2[52:63]))  # Mean motion (revs/day)
    e0 = jnp.array(float('0.' + tle_2[26:33].replace(' ', '0')))  # Eccentricity (not sure whether replace needed but it's in python sgp4)
    i0 = jnp.array(float(tle_2[8:16]))  # Inclination (degrees)
    w0 = jnp.array(float(tle_2[34:42]))  # Argument of perigee (degrees)
    Omega0 = jnp.array(float(tle_2[17:25]))  # Right ascension of the ascending node (degrees)
    M0 = jnp.array(float(tle_2[43:51]))  # Mean anomaly (degrees)
    epochdays = jnp.array(float(tle_1[20:32]))  # Epoch in days of year

    Bstar = jnp.array(float(tle_1[53] + '.' + tle_1[54:59]))  # Bstar mantissa
    bexp = int(tle_1[59:61])  # Exponent part of Bstar
    Bstar = Bstar * 10 ** bexp  # Drag coefficient (Earth radii^-1) 

    two_digit_year = int(tle_1[18:20])

    # can make this jax compatible later but not really needed since only done once per sat
    # Fix for years from 1957 to 2056
    if two_digit_year < 57:
        epochyr = 2000 + two_digit_year
    else:
        epochyr = 1900 + two_digit_year

    return Satellite(n0, e0, i0, w0, Omega0, M0, Bstar, epochdays, epochyr)


def tle2sat_array_fast(tle_1_array, tle_2_array):                             
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


# i think this needs fixing since I changed the Satellite class
# maybe change this in the future so tle2sat can handle arrays directly?
# should change this to take [(tle1_1, tle2_1), (tle1_2, tle2_2), ...] input format later since 
# this is what sgp4 package uses
def tle2sat_array(tle_1_array, tle_2_array):
    
    sats = [tle2sat(tle1, tle2) for tle1, tle2 in zip(tle_1_array, tle_2_array)]

    return Satellite(
        n0 = jnp.array([sat.n0 for sat in sats]),
        e0 = jnp.array([sat.e0 for sat in sats]),
        i0 = jnp.array([sat.i0 for sat in sats]),
        w0 = jnp.array([sat.w0 for sat in sats]),
        Omega0 = jnp.array([sat.Omega0 for sat in sats]),
        M0 = jnp.array([sat.M0 for sat in sats]),
        Bstar = jnp.array([sat.Bstar for sat in sats]),
        epochdays = jnp.array([sat.epochdays for sat in sats]),
        epochyr = jnp.array([sat.epochyr for sat in sats])
    )