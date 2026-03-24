from typing import NamedTuple

class Satellite(NamedTuple):
    n0: float           # Mean motion (revs/day)
    e0: float           # Eccentricity
    i0: float           # Inclination (degrees)
    w0: float           # Argument of perigee (degrees)
    Omega0: float       # Right ascension of the ascending node (degrees)
    M0: float           # Mean anomaly (degrees)
    Bstar: float        # Drag coefficient
    epochdays: float    # Epoch as fractional day of year (e.g. 13.5 = noon on Jan 13th)
    epochyr: float      # Epoch year (4-digit e.g. 2026)