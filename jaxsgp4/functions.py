"""High-level wrapper functions for SGP4 propagation."""

from .model import Satellite
from .propagation import sgp4
import jax.numpy as jnp

def sgp4_jdfr(sat: Satellite, jd, fr):
    """Propagate a satellite to a given Julian Date.

    Converts the target Julian Date to minutes-since-epoch (tsince)
    and calls sgp4().

    Note:
    In 32-bit precision (JAX default), the Julian Date epoch conversion can
    introduce errors of up to ~1 second due to limited floating-point
    resolution. For best accuracy in 32-bit mode, prefer sgp4() with tsince
    directly.

    Inputs:
      sat        : Satellite object containing orbital elements and parameters
      jd         : Julian Date 
      fr         : Fractional part of the day

    Returns:
      rv         : [x, y, z, vx, vy, vz] in km and km/s (TEME frame)
      error_code : 0 if no error (see sgp4 for codes)
    """
    # Convert satellite epoch to Julian Date
    # Formula: JD = 365*year + leap_day_correction + day_of_year + JD_offset
    # where 1721044.5 is the Julian Date of year 0, Jan 0.5 in this calendar scheme
    year = sat.epochyr
    days, fraction = jnp.divmod(sat.epochdays, 1.0)
    jd_epoch = year * 365 + (year - 1) // 4 + days + 1721044.5
    fr_epoch = jnp.round(fraction, 8) # round to match TLE precision (8 decimal places)

    tsince = (jd - jd_epoch) * 1440.0 + (fr - fr_epoch) * 1440.0

    return sgp4(sat, tsince)
