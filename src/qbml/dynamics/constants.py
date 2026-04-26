import numpy as np

from qbml import ureg, Q_

HBAR = 5308.8 * ureg.cm_1 * ureg.fs
C = 3e8 * ureg.m / ureg.s**2
K = 0.695 * ureg.cm_1 / ureg.kelvin
new_HBAR = HBAR.to('megahertz*us')

# What if ħ = 1?

unit_dict = {"time": "dimensionless",
             "energy": "dimensionless",
             "frequency": "dimensionless",
             "k": "dimensionless",
             "hbar": "dimensionless",
             "speed_of_light": "dimensionless"}

def get_constants(unit_dict : dict):
    """
    Return ħ, c, and kB for specified unit.
    """
    hbar = HBAR.to(f"{unit_dict['energy']} * {unit_dict['frequency']}")
    c = C
    kB = K
    return hbar, c, kB

