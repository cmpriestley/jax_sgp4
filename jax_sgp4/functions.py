# variations of the spg4 function to take different inputs etc. 
# (find a better way to name this later)

from .model import Satellite
from .propagation import sgp4
import jax.numpy as jnp
import jax

# # sgp4 function jitted and vectorised over times - for a single satellite propagated over many times
# # works for scalar and vector tsince inputs
# def jaxsgp4(sat: Satellite, tsince):
#     tsince = jnp.atleast_1d(tsince)
#     func = jax.jit(sgp4)
#     func = jax.vmap(func, in_axes=(None, 0))
#     result = func(sat, tsince)
#     return jnp.squeeze(result)

# alternative way to do the above function with separate scalar and vector functions
# probably get rid of this because it is slower than separate code and not much cleaner
def jaxsgp4(sat: Satellite, tsince):
    if jnp.ndim(tsince) == 0:
        return jaxsgp4_scalar(sat, tsince)
    return jaxsgp4_vector(sat, tsince)

# make two functions to compare the speed of this against vs doing sep functions for scalar or vector tince
jaxsgp4_scalar = jax.jit(sgp4)
# def jaxsgp4_scalar(sat: Satellite, tsince):
#     func = jax.jit(sgp4)
#     return func(sat, tsince)

# the same as old sgp4_many_times function
jaxsgp4_vector = jax.jit(jax.vmap(sgp4, in_axes=(None, 0)))
# def jaxsgp4_vector(sat: Satellite, tsince):
#     func = jax.jit(sgp4)
#     func = jax.vmap(func, in_axes=(None, 0))
#     return func(sat, tsince)

# remember need to jit compile this
def sgp4_jdfr(sat: Satellite, jd, fr):
    """
    SGP4 propagation algorithm using Julian Date and Fractional Day.
    
    Inputs:
      sat     : Satellite object containing orbital elements and parameters
      jd      : Julian Date (integer part)
      fr      : Fractional part of the day
    """

    # Calculate epoch in Julian Date and Fractional Day
    year = sat.epochyr
    days, fraction = jnp.divmod(sat.epochdays, 1.0)
    jd_epoch = year * 365 + (year - 1) // 4 + days + 1721044.5
    fr_epoch = jnp.round(fraction, 8) # round to match TLE precision

    tsince = (jd - jd_epoch) * 1440.0 + (fr - fr_epoch) * 1440.0
    rv = sgp4(sat, tsince)

    return rv

    # year mon day hr min sec to jd fr:
    # jd = (367.0 * year
    #      - 7 * (year + ((mon + 9) // 12.0)) * 0.25 // 1.0
	  #  + 275 * mon / 9.0 // 1.0
	  #  + day
    #      + 1721013.5)
    # fr = (sec + minute * 60.0 + hr * 3600.0) / 86400.0;
    # return jd, fr



# want sgp4_jdfr that does scalar, vector over jd, fr and vector over sats and vector over both?
# sgp4jdfr function jitted and vectorised over satellites
def jaxsgp4_jdfr(sat, jd, fr):
    return jax.jit(jax.vmap(sgp4_jdfr, in_axes=(0, None, None)))(sat, jd, fr)




# this fn is obsolete, can get rid of this 
def sgp4_many_times(sat: Satellite, tsince_array):
    """
    Vectorized SGP4 propagation over multiple times for a single satellite.
    
    Inputs:
      sat          : Satellite object containing orbital elements and parameters
      tsince_array : Array of times since epoch (minutes) jax or numpy both work?

    Returns:
      concatenated array of position and velocity vectors at each time in tsince_array (km and km/s)
    """

    jaxsgp4 = jax.jit(sgp4)

    # vectorize over tsince_array
    sgp4_vectorized = jax.vmap(jaxsgp4, in_axes=(None, 0))

    return sgp4_vectorized(sat, tsince_array)

def sgp4_many_sats(sat: Satellite, tsince):
    # fix what I named the function argument here
    """
    Vectorized SGP4 propagation over multiple satellites for a single time.

    Inputs:
      sat_array : Array of Satellite objects containing orbital elements and parameters
      tsince    : Time since epoch (minutes)

    Returns:
      concatenated array of position and velocity vectors for each satellite at time tsince (km and km/s)
    """
    
    jaxsgp4 = jax.jit(sgp4)

    # vectorize over sat_array
    sgp4_vectorized = jax.vmap(jaxsgp4, in_axes=(0, None))

    return sgp4_vectorized(sat, tsince)

# def jax sgp4 for tsince and jdfr inputs which vectorise and jit compile here later