import random
import numpy as np

from qbml.dynamics.hami import Hami
from qbml.dynamics.redfield import Redfield
from qbml.dynamics.spectraldensity import SplineSpecDen


def simulation(β: float, # has dimensions
               BETA: float, # no dimensions
               HBAR: float,
               qubit_frequency: float,
               TIMES: np.array,
               N_BATHS: int,
               SYS_HAMI: np.array,
               SB_HAMI: np.array,
               ρ_0: np.array,
               ) -> np.array:
    SPECDEN = []

    # Run dynamics.
    HAMI = Hami(SYS_HAMI, 0, SB_HAMI, SPECDEN, TIMES[1])
    r = Redfield(HAMI, BETA, TIMES, hbar=HBAR)

    rdm = np.array(r.propagate(ρ_0))
    sigma_x = np.expand_dims(rdm[:, 1] + rdm[:, 2], axis=-1)
    sigma_y = np.expand_dims(1j * rdm[:, 1] - 1j * rdm[:, 2], axis=-1)
    sigma_z = np.expand_dims(rdm[:, 0] - rdm[:, 3], axis=-1)
    spins_t = np.concatenate((sigma_x, sigma_y, sigma_z), axis=-1)
    return spins_t, SPECDEN
